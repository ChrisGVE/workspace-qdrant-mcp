use super::*;
use std::io::Write;
use tempfile::NamedTempFile;

use super::super::extraction::{
    count_docx_images, extract_csv, extract_jupyter, extract_spreadsheet,
};

// --- New format detection tests ---

#[test]
fn test_detect_document_type_spreadsheet_formats() {
    assert_eq!(
        detect_document_type(Path::new("data.xlsx")),
        DocumentType::Xlsx
    );
    assert_eq!(
        detect_document_type(Path::new("data.xls")),
        DocumentType::Xls
    );
    assert_eq!(
        detect_document_type(Path::new("DATA.XLSX")),
        DocumentType::Xlsx
    );
    assert_eq!(
        detect_document_type(Path::new("report.XLS")),
        DocumentType::Xls
    );
}

#[test]
fn test_detect_document_type_csv() {
    assert_eq!(
        detect_document_type(Path::new("data.csv")),
        DocumentType::Csv
    );
    assert_eq!(
        detect_document_type(Path::new("data.tsv")),
        DocumentType::Csv
    );
    assert_eq!(
        detect_document_type(Path::new("DATA.CSV")),
        DocumentType::Csv
    );
}

#[test]
fn test_detect_document_type_jupyter() {
    assert_eq!(
        detect_document_type(Path::new("notebook.ipynb")),
        DocumentType::Jupyter
    );
    assert_eq!(
        detect_document_type(Path::new("analysis.IPYNB")),
        DocumentType::Jupyter
    );
}

#[test]
fn test_detect_document_type_web_content() {
    // HTML files now get language metadata instead of being treated as plain text
    assert_eq!(
        detect_document_type(Path::new("index.html")),
        DocumentType::Code("html".to_string())
    );
    assert_eq!(
        detect_document_type(Path::new("page.htm")),
        DocumentType::Code("html".to_string())
    );
    assert_eq!(
        detect_document_type(Path::new("doc.xhtml")),
        DocumentType::Code("html".to_string())
    );
    assert_eq!(
        detect_document_type(Path::new("data.xml")),
        DocumentType::Code("xml".to_string())
    );
    assert_eq!(
        detect_document_type(Path::new("icon.svg")),
        DocumentType::Code("xml".to_string())
    );
}

#[test]
fn test_detect_document_type_new_languages() {
    // PowerShell
    assert_eq!(
        detect_document_type(Path::new("script.ps1")),
        DocumentType::Code("powershell".to_string())
    );
    assert_eq!(
        detect_document_type(Path::new("module.psm1")),
        DocumentType::Code("powershell".to_string())
    );
    // D language
    assert_eq!(
        detect_document_type(Path::new("main.d")),
        DocumentType::Code("d".to_string())
    );
    // Zig
    assert_eq!(
        detect_document_type(Path::new("build.zig")),
        DocumentType::Code("zig".to_string())
    );
    // Dart
    assert_eq!(
        detect_document_type(Path::new("app.dart")),
        DocumentType::Code("dart".to_string())
    );
    // Protocol Buffers
    assert_eq!(
        detect_document_type(Path::new("service.proto")),
        DocumentType::Code("protobuf".to_string())
    );
    // GraphQL
    assert_eq!(
        detect_document_type(Path::new("schema.graphql")),
        DocumentType::Code("graphql".to_string())
    );
    assert_eq!(
        detect_document_type(Path::new("query.gql")),
        DocumentType::Code("graphql".to_string())
    );
    // Astro
    assert_eq!(
        detect_document_type(Path::new("page.astro")),
        DocumentType::Code("astro".to_string())
    );
}

#[test]
fn test_detect_document_type_compound_extensions() {
    // .d.ts should be TypeScript, not D language
    assert_eq!(
        detect_document_type(Path::new("types.d.ts")),
        DocumentType::Code("typescript".to_string())
    );
    assert_eq!(
        detect_document_type(Path::new("module.d.mts")),
        DocumentType::Code("typescript".to_string())
    );
    assert_eq!(
        detect_document_type(Path::new("common.d.cts")),
        DocumentType::Code("typescript".to_string())
    );
    // But plain .d should be D language
    assert_eq!(
        detect_document_type(Path::new("main.d")),
        DocumentType::Code("d".to_string())
    );
}

#[test]
fn test_detect_document_type_text_extensions() {
    // rst, org, adoc are plain text
    assert_eq!(
        detect_document_type(Path::new("doc.rst")),
        DocumentType::Text
    );
    assert_eq!(
        detect_document_type(Path::new("notes.org")),
        DocumentType::Text
    );
    assert_eq!(
        detect_document_type(Path::new("guide.adoc")),
        DocumentType::Text
    );
}

// --- CSV extraction tests ---

#[test]
fn test_extract_csv_basic() {
    let mut tmp = NamedTempFile::with_suffix(".csv").unwrap();
    write!(tmp, "name,age,city\nAlice,30,New York\nBob,25,London\n").unwrap();
    let result = extract_csv(tmp.path());
    assert!(result.is_ok());
    let (text, metadata) = result.unwrap();
    assert!(text.contains("name"));
    assert!(text.contains("Alice"));
    assert!(text.contains("Bob"));
    assert_eq!(metadata.get("source_format").unwrap(), "csv");
    assert_eq!(metadata.get("row_count").unwrap(), "2");
    assert_eq!(metadata.get("column_count").unwrap(), "3");
}

#[test]
fn test_extract_csv_tsv() {
    let mut tmp = NamedTempFile::with_suffix(".tsv").unwrap();
    write!(tmp, "col1\tcol2\nval1\tval2\n").unwrap();
    let result = extract_csv(tmp.path());
    assert!(result.is_ok());
    let (text, metadata) = result.unwrap();
    assert!(text.contains("col1"));
    assert!(text.contains("val1"));
    assert_eq!(metadata.get("source_format").unwrap(), "tsv");
}

#[test]
fn test_extract_csv_empty() {
    let mut tmp = NamedTempFile::with_suffix(".csv").unwrap();
    write!(tmp, "").unwrap();
    let result = extract_csv(tmp.path());
    assert!(result.is_err());
}

// --- Jupyter extraction tests ---

#[test]
fn test_extract_jupyter_basic() {
    let notebook = serde_json::json!({
        "metadata": {
            "kernelspec": { "language": "python" }
        },
        "cells": [
            {
                "cell_type": "markdown",
                "source": ["# Test Notebook\n", "This is a test."]
            },
            {
                "cell_type": "code",
                "source": ["import pandas as pd\n", "df = pd.read_csv('data.csv')"]
            }
        ],
        "nbformat": 4,
        "nbformat_minor": 2
    })
    .to_string();

    let mut tmp = NamedTempFile::with_suffix(".ipynb").unwrap();
    write!(tmp, "{}", &notebook).unwrap();
    let result = extract_jupyter(tmp.path());
    assert!(result.is_ok());
    let (text, metadata) = result.unwrap();
    assert!(text.contains("Test Notebook"));
    assert!(text.contains("import pandas"));
    assert_eq!(metadata.get("language").unwrap(), "python");
    assert_eq!(metadata.get("cell_count").unwrap(), "2");
    assert_eq!(metadata.get("code_cells").unwrap(), "1");
    assert_eq!(metadata.get("markdown_cells").unwrap(), "1");
}

#[test]
fn test_extract_jupyter_source_as_string() {
    // Some notebooks have source as a single string instead of array
    let notebook = serde_json::json!({
        "metadata": { "language_info": { "name": "r" } },
        "cells": [
            { "cell_type": "code", "source": "x <- 1:10\nplot(x)" }
        ],
        "nbformat": 4,
        "nbformat_minor": 2
    })
    .to_string();

    let mut tmp = NamedTempFile::with_suffix(".ipynb").unwrap();
    write!(tmp, "{}", &notebook).unwrap();
    let result = extract_jupyter(tmp.path());
    assert!(result.is_ok());
    let (text, metadata) = result.unwrap();
    assert!(text.contains("x <- 1:10"));
    assert_eq!(metadata.get("language").unwrap(), "r");
}

#[test]
fn test_extract_jupyter_invalid_json() {
    let mut tmp = NamedTempFile::with_suffix(".ipynb").unwrap();
    write!(tmp, "not valid json").unwrap();
    let result = extract_jupyter(tmp.path());
    assert!(result.is_err());
}

#[test]
fn test_extract_jupyter_no_cells() {
    let notebook = serde_json::json!({ "metadata": {}, "nbformat": 4 }).to_string();
    let mut tmp = NamedTempFile::with_suffix(".ipynb").unwrap();
    write!(tmp, "{}", &notebook).unwrap();
    let result = extract_jupyter(tmp.path());
    assert!(result.is_err());
}

// --- Spreadsheet extraction tests ---

#[test]
fn test_extract_spreadsheet_invalid_file() {
    let mut tmp = NamedTempFile::with_suffix(".xlsx").unwrap();
    write!(tmp, "not a valid xlsx file").unwrap();
    let result = extract_spreadsheet(tmp.path());
    assert!(result.is_err());
}

// --- Allowed extensions tests ---

#[test]
fn test_allowed_extensions_new_formats() {
    use crate::allowed_extensions::AllowedExtensions;
    let ae = AllowedExtensions::default();
    // Project extensions
    assert!(ae.is_allowed("data.csv", "projects"));
    assert!(ae.is_allowed("data.tsv", "projects"));
    assert!(ae.is_allowed("notebook.ipynb", "projects"));
    // Library extensions
    assert!(ae.is_allowed("report.xlsx", "libraries"));
    assert!(ae.is_allowed("legacy.xls", "libraries"));
}

// --- Image counting tests (DOCX only; PDF image counting removed with lopdf) ---

#[test]
fn test_count_docx_images_with_media() {
    use std::fs::File;
    // Build a minimal DOCX ZIP with images in word/media/
    let temp = NamedTempFile::new().unwrap();
    {
        let mut zip = zip::ZipWriter::new(&temp);
        let options = zip::write::SimpleFileOptions::default();
        zip.start_file("word/document.xml", options).unwrap();
        zip.write_all(b"<w:document/>").unwrap();
        zip.start_file("word/media/image1.png", options).unwrap();
        zip.write_all(b"PNG_DATA").unwrap();
        zip.start_file("word/media/image2.jpeg", options).unwrap();
        zip.write_all(b"JPEG_DATA").unwrap();
        zip.start_file("word/media/chart1.xml", options).unwrap();
        zip.write_all(b"<chart/>").unwrap();
        zip.finish().unwrap();
    }

    let file = File::open(temp.path()).unwrap();
    let archive = zip::ZipArchive::new(file).unwrap();
    assert_eq!(count_docx_images(&archive), 2);
}

#[test]
fn test_count_docx_images_no_media() {
    use std::fs::File;
    let temp = NamedTempFile::new().unwrap();
    {
        let mut zip = zip::ZipWriter::new(&temp);
        let options = zip::write::SimpleFileOptions::default();
        zip.start_file("word/document.xml", options).unwrap();
        zip.write_all(b"<w:document/>").unwrap();
        zip.finish().unwrap();
    }

    let file = File::open(temp.path()).unwrap();
    let archive = zip::ZipArchive::new(file).unwrap();
    assert_eq!(count_docx_images(&archive), 0);
}

#[cfg(feature = "ocr")]
mod ocr_integration_tests {
    use super::*;
    use crate::document_processor::ocr;
    use crate::ocr::{OcrConfig, OcrEngine};

    #[test]
    fn test_enrich_text_with_ocr_no_engine() {
        let temp = NamedTempFile::with_suffix(".txt").unwrap();
        std::fs::write(temp.path(), "hello").unwrap();
        let mut metadata = HashMap::new();
        let result = ocr::enrich_text_with_ocr(
            temp.path(),
            "original text".to_string(),
            &mut metadata,
            None,
        );
        assert_eq!(result, "original text");
    }

    #[test]
    fn test_enrich_text_with_ocr_no_images() {
        let temp = NamedTempFile::with_suffix(".txt").unwrap();
        std::fs::write(temp.path(), "hello world").unwrap();

        let config = OcrConfig::default();
        if !config.tessdata_path.exists() {
            return; // Skip without Tesseract
        }
        let engine = match OcrEngine::new(&config) {
            Ok(e) => e,
            Err(_) => return,
        };

        let mut metadata = HashMap::new();
        let result = ocr::enrich_text_with_ocr(
            temp.path(),
            "original text".to_string(),
            &mut metadata,
            Some(&engine),
        );
        assert_eq!(result, "original text");
        assert_eq!(metadata.get("images_detected").unwrap(), "0");
    }

    #[test]
    fn test_with_ocr_constructor() {
        let config = OcrConfig::default();
        if !config.tessdata_path.exists() {
            return;
        }
        let engine = match OcrEngine::new(&config) {
            Ok(e) => e,
            Err(_) => return,
        };

        let processor = DocumentProcessor::with_ocr(ChunkingConfig::default(), engine);
        assert!(processor.ocr_engine.is_some());
    }

    #[test]
    fn test_processor_without_ocr_has_none() {
        let processor = DocumentProcessor::new();
        assert!(processor.ocr_engine.is_none());
    }
}

// --- Binary-content gate (regression for the memexd OOM on a Mach-O fixture) ---

/// A file containing NUL bytes must be rejected as binary, never lossy-decoded
/// into garbage "text" and fed to the chunker.
#[test]
fn test_extract_text_rejects_binary_nul_bytes() {
    use super::super::extraction::extract_text_with_encoding;
    use super::super::types::DocumentProcessorError;

    // Mach-O-like header: magic + NUL padding (exactly what `bookshelf` looked like).
    let mut tmp = NamedTempFile::new().unwrap();
    let mut buf = vec![0xCF, 0xFA, 0xED, 0xFE];
    buf.extend(std::iter::repeat(0u8).take(64));
    buf.extend_from_slice(b"__TEXT__text");
    tmp.write_all(&buf).unwrap();
    tmp.flush().unwrap();

    let err = extract_text_with_encoding(tmp.path()).unwrap_err();
    assert!(
        matches!(err, DocumentProcessorError::BinaryFile(_)),
        "expected BinaryFile, got {err:?}"
    );
}

/// Plain UTF-8 text (no NUL) must still extract cleanly.
#[test]
fn test_extract_text_accepts_utf8() {
    use super::super::extraction::extract_text_with_encoding;

    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all("fn main() {}\nlet x = 1;\n".as_bytes()).unwrap();
    tmp.flush().unwrap();

    let (text, meta) = extract_text_with_encoding(tmp.path()).unwrap();
    assert!(text.contains("fn main"));
    assert_eq!(meta.get("encoding").map(String::as_str), Some("utf-8"));
}

/// UTF-16 LE text carries NUL bytes but a BOM — the gate must NOT reject it.
#[test]
fn test_extract_text_utf16_bom_not_treated_as_binary() {
    use super::super::extraction::extract_text_with_encoding;

    let mut tmp = NamedTempFile::new().unwrap();
    let mut buf = vec![0xFF, 0xFE]; // UTF-16 LE BOM
    for b in "hello".encode_utf16() {
        buf.extend_from_slice(&b.to_le_bytes());
    }
    tmp.write_all(&buf).unwrap();
    tmp.flush().unwrap();

    // Must not be a BinaryFile rejection (decodes via the encoding path).
    let res = extract_text_with_encoding(tmp.path());
    assert!(
        res.is_ok(),
        "UTF-16 BOM text wrongly rejected: {:?}",
        res.err()
    );
}
