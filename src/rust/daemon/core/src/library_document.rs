//! Library document structural splitting and token-based chunking.
//!
//! Splits documents into structural units (pages, slides, sections, chapters)
//! and chunks each unit into token-budgeted pieces. Creates parent records
//! (no vectors) for each structural unit and child chunk records that reference
//! their parent.
//!
//! Two document families:
//! - **Page-based** (Task 7): PDF, DOCX, PPTX, ODP, ODT, ODS, RTF
//! - **Stream-based** (Task 8): EPUB, HTML, Markdown, plain text

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use thiserror::Error;
use tracing::debug;

use crate::parent_unit::{
    self, ParentUnitRecord, UNIT_TYPE_DOCX_SECTION, UNIT_TYPE_EPUB_SECTION, UNIT_TYPE_PDF_PAGE,
    UNIT_TYPE_TEXT_SECTION,
};
use crate::tokenizer::{ModelTokenizer, TokenizerError};

/// Errors from library document processing
#[derive(Error, Debug)]
pub enum LibraryDocumentError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Extraction error ({format}): {message}")]
    Extraction { format: String, message: String },

    #[error("Tokenizer error: {0}")]
    Tokenizer(#[from] TokenizerError),

    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),
}

/// A structural unit extracted from a document (page, slide, section, chapter).
#[derive(Debug, Clone)]
pub struct StructuralUnit {
    /// Unit type for Qdrant payload (pdf_page, epub_section, etc.)
    pub unit_type: String,
    /// Locator JSON for this unit within the document
    pub unit_locator: serde_json::Value,
    /// Full text content of this unit
    pub text: String,
    /// Optional title for this unit (slide title, chapter title, heading)
    pub title: Option<String>,
}

/// A child chunk with its parent reference.
#[derive(Debug, Clone)]
pub struct LibraryChunk {
    /// The chunk text (raw, without header)
    pub text_raw: String,
    /// The chunk text with header prepended (for indexing/embedding)
    pub text_indexed: String,
    /// Number of tokens in text_indexed
    pub token_count: usize,
    /// Character start offset within the parent unit's text
    pub char_start: usize,
    /// Character end offset within the parent unit's text
    pub char_end: usize,
    /// Chunk index within this parent unit
    pub chunk_index: usize,
    /// Point ID of the parent unit record
    pub parent_unit_id: String,
}

/// Result of processing a single structural unit.
#[derive(Debug, Clone)]
pub struct ProcessedUnit {
    /// The parent record (no vectors)
    pub parent: ParentUnitRecord,
    /// Token-based child chunks
    pub chunks: Vec<LibraryChunk>,
}

/// Result of processing an entire library document.
#[derive(Debug, Clone)]
pub struct LibraryDocumentResult {
    /// All processed units (parents + their chunks)
    pub units: Vec<ProcessedUnit>,
    /// Document metadata extracted during processing
    pub metadata: HashMap<String, String>,
    /// Total chunks across all units
    pub total_chunks: usize,
}

// ─── Page-based splitting (Task 7) ──────────────────────────────────────────

/// Split a PDF into per-page structural units.
///
/// Uses `pdf-extract` which inserts form feed (`\x0C`) characters between pages.
/// Falls back to paragraph-based splitting if no form feeds are found.
pub fn split_pdf(file_path: &Path) -> Result<Vec<StructuralUnit>, LibraryDocumentError> {
    let path_buf = file_path.to_path_buf();
    let result = std::panic::catch_unwind(|| pdf_extract::extract_text(&path_buf));

    let full_text = match result {
        Ok(Ok(text)) => text,
        Ok(Err(e)) => {
            return Err(LibraryDocumentError::Extraction {
                format: "pdf".into(),
                message: e.to_string(),
            })
        }
        Err(_panic) => {
            return Err(LibraryDocumentError::Extraction {
                format: "pdf".into(),
                message: format!(
                    "PDF parsing panicked (likely malformed font encoding): {}",
                    file_path.display()
                ),
            })
        }
    };

    // Split by form feed characters (inserted between pages by pdf-extract)
    let pages: Vec<&str> = full_text.split('\x0C').collect();

    if pages.len() <= 1 {
        // No form feeds — treat entire document as a single page
        let cleaned = clean_text(&full_text);
        if cleaned.is_empty() {
            return Ok(vec![]);
        }
        return Ok(vec![StructuralUnit {
            unit_type: UNIT_TYPE_PDF_PAGE.to_string(),
            unit_locator: serde_json::json!({"page": 1}),
            text: cleaned,
            title: None,
        }]);
    }

    let mut units = Vec::new();
    for (i, page_text) in pages.iter().enumerate() {
        let cleaned = clean_text(page_text);
        if cleaned.is_empty() {
            continue;
        }
        units.push(StructuralUnit {
            unit_type: UNIT_TYPE_PDF_PAGE.to_string(),
            unit_locator: serde_json::json!({"page": i + 1}),
            text: cleaned,
            title: None,
        });
    }

    debug!("PDF split into {} pages", units.len());
    Ok(units)
}

/// Split a PPTX into per-slide structural units.
pub fn split_pptx(file_path: &Path) -> Result<Vec<StructuralUnit>, LibraryDocumentError> {
    let file = File::open(file_path)?;
    let mut archive = zip::ZipArchive::new(file).map_err(|e| LibraryDocumentError::Extraction {
        format: "pptx".into(),
        message: e.to_string(),
    })?;

    // Collect and sort slide file names
    let mut slide_names: Vec<String> = (0..archive.len())
        .filter_map(|i| {
            archive.by_index(i).ok().and_then(|f| {
                let name = f.name().to_string();
                if name.starts_with("ppt/slides/slide") && name.ends_with(".xml") {
                    Some(name)
                } else {
                    None
                }
            })
        })
        .collect();
    slide_names.sort();

    let mut units = Vec::new();
    for (i, slide_name) in slide_names.iter().enumerate() {
        if let Ok(mut slide_file) = archive.by_name(slide_name) {
            let mut content = String::new();
            slide_file.read_to_string(&mut content)?;
            let slide_text = extract_text_from_xml_tags(&content, "a:t");
            let cleaned = clean_text(&slide_text);
            if cleaned.is_empty() {
                continue;
            }
            units.push(StructuralUnit {
                unit_type: "pptx_slide".to_string(),
                unit_locator: serde_json::json!({"slide": i + 1}),
                text: cleaned,
                title: None, // Could extract from first a:t in sp with type="title"
            });
        }
    }

    debug!("PPTX split into {} slides", units.len());
    Ok(units)
}

/// Split a DOCX into section-level structural units.
///
/// Splits at `w:sectPr` (section properties) boundaries in the XML.
/// If no section boundaries found, treats entire document as one section.
pub fn split_docx(file_path: &Path) -> Result<Vec<StructuralUnit>, LibraryDocumentError> {
    let file = File::open(file_path)?;
    let mut archive = zip::ZipArchive::new(file).map_err(|e| LibraryDocumentError::Extraction {
        format: "docx".into(),
        message: e.to_string(),
    })?;

    let mut xml_content = String::new();
    if let Ok(mut doc_file) = archive.by_name("word/document.xml") {
        doc_file.read_to_string(&mut xml_content)?;
    } else {
        return Err(LibraryDocumentError::Extraction {
            format: "docx".into(),
            message: "No word/document.xml found".into(),
        });
    }

    // Split XML by section properties (w:sectPr marks end of a section)
    let sections: Vec<&str> = xml_content.split("w:sectPr").collect();

    let mut units = Vec::new();
    for (i, section_xml) in sections.iter().enumerate() {
        // Last fragment after final sectPr is usually empty/closing tags
        if i == sections.len() - 1 && sections.len() > 1 {
            continue;
        }
        let text = extract_docx_text_from_xml(section_xml);
        let cleaned = clean_text(&text);
        if cleaned.is_empty() {
            continue;
        }
        units.push(StructuralUnit {
            unit_type: UNIT_TYPE_DOCX_SECTION.to_string(),
            unit_locator: serde_json::json!({"section": i + 1}),
            text: cleaned,
            title: None,
        });
    }

    // If no sections found, treat entire document as one section
    if units.is_empty() {
        let full_text = extract_docx_text_from_xml(&xml_content);
        let cleaned = clean_text(&full_text);
        if !cleaned.is_empty() {
            units.push(StructuralUnit {
                unit_type: UNIT_TYPE_DOCX_SECTION.to_string(),
                unit_locator: serde_json::json!({"section": 1}),
                text: cleaned,
                title: None,
            });
        }
    }

    debug!("DOCX split into {} sections", units.len());
    Ok(units)
}

/// Split an ODP (OpenDocument Presentation) into per-slide structural units.
pub fn split_odp(file_path: &Path) -> Result<Vec<StructuralUnit>, LibraryDocumentError> {
    split_opendocument_pages(file_path, "odp", "draw:page", "odp_slide", "slide")
}

/// Split an ODT (OpenDocument Text) into section-level structural units.
///
/// Splits by `text:section` elements, or falls back to treating the whole document as one section.
pub fn split_odt(file_path: &Path) -> Result<Vec<StructuralUnit>, LibraryDocumentError> {
    let file = File::open(file_path)?;
    let mut archive = zip::ZipArchive::new(file).map_err(|e| LibraryDocumentError::Extraction {
        format: "odt".into(),
        message: e.to_string(),
    })?;

    let mut xml_content = String::new();
    if let Ok(mut content_file) = archive.by_name("content.xml") {
        content_file.read_to_string(&mut xml_content)?;
    } else {
        return Err(LibraryDocumentError::Extraction {
            format: "odt".into(),
            message: "No content.xml found".into(),
        });
    }

    // Try splitting by text:section elements
    let sections: Vec<&str> = xml_content.split("<text:section").collect();

    let mut units = Vec::new();
    if sections.len() > 1 {
        // Skip first element (before the first section)
        for (i, section_xml) in sections.iter().skip(1).enumerate() {
            let text = extract_text_from_xml_tags(section_xml, "text:p");
            let cleaned = clean_text(&text);
            if cleaned.is_empty() {
                continue;
            }
            units.push(StructuralUnit {
                unit_type: "odt_section".to_string(),
                unit_locator: serde_json::json!({"section": i + 1}),
                text: cleaned,
                title: None,
            });
        }
    }

    // If no sections, treat entire document as one section
    if units.is_empty() {
        let text = extract_text_from_xml_tags(&xml_content, "text:p");
        let cleaned = clean_text(&text);
        if !cleaned.is_empty() {
            units.push(StructuralUnit {
                unit_type: "odt_section".to_string(),
                unit_locator: serde_json::json!({"section": 1}),
                text: cleaned,
                title: None,
            });
        }
    }

    debug!("ODT split into {} sections", units.len());
    Ok(units)
}

/// Split an ODS (OpenDocument Spreadsheet) into per-sheet structural units.
pub fn split_ods(file_path: &Path) -> Result<Vec<StructuralUnit>, LibraryDocumentError> {
    let file = File::open(file_path)?;
    let mut archive = zip::ZipArchive::new(file).map_err(|e| LibraryDocumentError::Extraction {
        format: "ods".into(),
        message: e.to_string(),
    })?;

    let mut xml_content = String::new();
    if let Ok(mut content_file) = archive.by_name("content.xml") {
        content_file.read_to_string(&mut xml_content)?;
    } else {
        return Err(LibraryDocumentError::Extraction {
            format: "ods".into(),
            message: "No content.xml found".into(),
        });
    }

    // Split by table:table elements
    let tables: Vec<&str> = xml_content.split("<table:table ").collect();

    let mut units = Vec::new();
    for (i, table_xml) in tables.iter().skip(1).enumerate() {
        // Extract sheet name from table:name attribute
        let sheet_name = extract_xml_attr(table_xml, "table:name").unwrap_or_else(|| format!("Sheet{}", i + 1));

        // Extract text from table cells
        let text = extract_text_from_xml_tags(table_xml, "text:p");
        let cleaned = clean_text(&text);
        if cleaned.is_empty() {
            continue;
        }
        units.push(StructuralUnit {
            unit_type: "ods_sheet".to_string(),
            unit_locator: serde_json::json!({"sheet": sheet_name}),
            text: cleaned,
            title: Some(sheet_name),
        });
    }

    debug!("ODS split into {} sheets", units.len());
    Ok(units)
}

/// Split an RTF file into a single structural unit.
///
/// RTF has no page/section structure accessible without a full layout engine,
/// so the entire content is treated as a single section.
pub fn split_rtf(file_path: &Path) -> Result<Vec<StructuralUnit>, LibraryDocumentError> {
    let mut file = File::open(file_path)?;
    let mut raw = String::new();
    file.read_to_string(&mut raw)?;

    let text = strip_rtf_control_codes(&raw);
    let cleaned = clean_text(&text);
    if cleaned.is_empty() {
        return Ok(vec![]);
    }

    Ok(vec![StructuralUnit {
        unit_type: UNIT_TYPE_TEXT_SECTION.to_string(),
        unit_locator: serde_json::json!({"section": 1}),
        text: cleaned,
        title: None,
    }])
}

// ─── Stream-based splitting (Task 8) ────────────────────────────────────────

/// Split an EPUB into per-chapter structural units.
pub fn split_epub(file_path: &Path) -> Result<Vec<StructuralUnit>, LibraryDocumentError> {
    let mut doc = epub::doc::EpubDoc::new(file_path).map_err(|e| {
        LibraryDocumentError::Extraction {
            format: "epub".into(),
            message: e.to_string(),
        }
    })?;

    let mut units = Vec::new();
    let mut spine_index = 0usize;

    loop {
        let spine_id = doc
            .get_current_id()
            .unwrap_or_else(|| format!("spine_{}", spine_index));

        if let Some((content, _mime)) = doc.get_current_str() {
            let text = html2text::from_read(content.as_bytes(), 80);
            let cleaned = clean_text(&text);

            if !cleaned.is_empty() {
                // Try to extract chapter title from HTML content
                let chapter_title = extract_html_heading(&content);

                let mut locator = serde_json::json!({"spine_id": spine_id});
                if let Some(ref title) = chapter_title {
                    locator["chapter_title"] =
                        serde_json::Value::String(title.clone());
                }

                units.push(StructuralUnit {
                    unit_type: UNIT_TYPE_EPUB_SECTION.to_string(),
                    unit_locator: locator,
                    text: cleaned,
                    title: chapter_title,
                });
            }
        }

        spine_index += 1;
        if !doc.go_next() {
            break;
        }
    }

    debug!("EPUB split into {} chapters", units.len());
    Ok(units)
}

/// Split an HTML document into heading-based structural units.
pub fn split_html(file_path: &Path) -> Result<Vec<StructuralUnit>, LibraryDocumentError> {
    let mut file = File::open(file_path)?;
    let mut raw = String::new();
    file.read_to_string(&mut raw)?;

    // Convert HTML to plain text first
    let full_text = html2text::from_read(raw.as_bytes(), 120);

    // Split by heading-like patterns in the text output
    // html2text renders headings with emphasis markers or on their own lines
    split_text_by_headings(&full_text, "html_section")
}

/// Split a Markdown document into heading-based structural units.
pub fn split_markdown(file_path: &Path) -> Result<Vec<StructuralUnit>, LibraryDocumentError> {
    let mut file = File::open(file_path)?;
    let mut raw = String::new();
    file.read_to_string(&mut raw)?;

    split_markdown_text(&raw)
}

/// Split markdown text into heading-based sections.
///
/// Recognizes ATX-style headings (# through ######).
/// YAML frontmatter is skipped.
fn split_markdown_text(text: &str) -> Result<Vec<StructuralUnit>, LibraryDocumentError> {
    let mut units = Vec::new();
    let mut current_title: Option<String> = None;
    let mut current_level = 0usize;
    let mut current_text = String::new();
    let mut section_index = 0usize;
    let mut in_frontmatter = false;
    for (line_idx, line) in text.lines().enumerate() {
        // Handle YAML frontmatter
        if line_idx == 0 && line.trim() == "---" {
            in_frontmatter = true;
            continue;
        }
        if in_frontmatter {
            if line.trim() == "---" || line.trim() == "..." {
                in_frontmatter = false;
            }
            continue;
        }

        // Check for ATX headings
        if let Some(heading) = parse_atx_heading(line) {
            // Flush previous section
            if !current_text.is_empty() || current_title.is_some() {
                let cleaned = clean_text(&current_text);
                if !cleaned.is_empty() {
                    let mut locator = serde_json::json!({
                        "section_index": section_index,
                        "heading_level": current_level,
                    });
                    if let Some(ref title) = current_title {
                        locator["title"] = serde_json::Value::String(title.clone());
                    }
                    units.push(StructuralUnit {
                        unit_type: "markdown_section".to_string(),
                        unit_locator: locator,
                        text: cleaned,
                        title: current_title.take(),
                    });
                    section_index += 1;
                }
            }
            current_title = Some(heading.title);
            current_level = heading.level;
            current_text.clear();
        } else {
            if !current_text.is_empty() || !line.trim().is_empty() {
                if !current_text.is_empty() {
                    current_text.push('\n');
                }
                current_text.push_str(line);
            }
        }
    }

    // Flush final section
    let cleaned = clean_text(&current_text);
    if !cleaned.is_empty() || current_title.is_some() {
        let actual_text = if cleaned.is_empty() {
            current_title.clone().unwrap_or_default()
        } else {
            cleaned
        };
        if !actual_text.is_empty() {
            // If we never encountered any headings, use text_section type
            let unit_type = if section_index == 0 && current_title.is_none() {
                UNIT_TYPE_TEXT_SECTION.to_string()
            } else {
                "markdown_section".to_string()
            };
            let mut locator = serde_json::json!({
                "section_index": section_index,
                "heading_level": current_level,
            });
            if let Some(ref title) = current_title {
                locator["title"] = serde_json::Value::String(title.clone());
            }
            units.push(StructuralUnit {
                unit_type,
                unit_locator: locator,
                text: actual_text,
                title: current_title,
            });
        }
    }

    debug!("Markdown split into {} sections", units.len());
    Ok(units)
}

/// Split a plain text file into paragraph-group structural units.
///
/// Groups paragraphs (separated by double newlines) into sections of
/// reasonable size. Very short documents become a single section.
pub fn split_plain_text(file_path: &Path) -> Result<Vec<StructuralUnit>, LibraryDocumentError> {
    let mut file = File::open(file_path)?;
    let mut raw = String::new();
    file.read_to_string(&mut raw)?;

    split_text_by_paragraphs(&raw)
}

/// Split text into paragraph-group sections.
///
/// Groups consecutive paragraphs into sections. A new section starts
/// when accumulated text exceeds ~2000 characters.
fn split_text_by_paragraphs(text: &str) -> Result<Vec<StructuralUnit>, LibraryDocumentError> {
    let paragraphs: Vec<&str> = text.split("\n\n").collect();

    if paragraphs.len() <= 1 {
        let cleaned = clean_text(text);
        if cleaned.is_empty() {
            return Ok(vec![]);
        }
        return Ok(vec![StructuralUnit {
            unit_type: UNIT_TYPE_TEXT_SECTION.to_string(),
            unit_locator: serde_json::json!({"section_index": 0}),
            text: cleaned,
            title: None,
        }]);
    }

    let mut units = Vec::new();
    let mut current = String::new();
    let mut section_index = 0usize;
    let section_target_chars = 2000;

    for para in &paragraphs {
        let trimmed = para.trim();
        if trimmed.is_empty() {
            continue;
        }

        if !current.is_empty() && current.len() + trimmed.len() > section_target_chars {
            let cleaned = clean_text(&current);
            if !cleaned.is_empty() {
                units.push(StructuralUnit {
                    unit_type: UNIT_TYPE_TEXT_SECTION.to_string(),
                    unit_locator: serde_json::json!({"section_index": section_index}),
                    text: cleaned,
                    title: Some(format!("Section {}", section_index + 1)),
                });
                section_index += 1;
            }
            current.clear();
        }

        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(trimmed);
    }

    // Flush remaining
    let cleaned = clean_text(&current);
    if !cleaned.is_empty() {
        units.push(StructuralUnit {
            unit_type: UNIT_TYPE_TEXT_SECTION.to_string(),
            unit_locator: serde_json::json!({"section_index": section_index}),
            text: cleaned,
            title: if section_index > 0 {
                Some(format!("Section {}", section_index + 1))
            } else {
                None
            },
        });
    }

    debug!("Text split into {} sections", units.len());
    Ok(units)
}

// ─── Unified processing ─────────────────────────────────────────────────────

/// Split a document into structural units based on its format.
///
/// Dispatches to the appropriate format-specific splitter.
pub fn split_document(
    file_path: &Path,
    source_format: &str,
) -> Result<Vec<StructuralUnit>, LibraryDocumentError> {
    match source_format {
        "pdf" => split_pdf(file_path),
        "pptx" => split_pptx(file_path),
        "docx" => split_docx(file_path),
        "odp" => split_odp(file_path),
        "odt" => split_odt(file_path),
        "ods" => split_ods(file_path),
        "rtf" => split_rtf(file_path),
        "epub" => split_epub(file_path),
        "html" => split_html(file_path),
        "markdown" | "md" => split_markdown(file_path),
        "text" | "txt" => split_plain_text(file_path),
        other => Err(LibraryDocumentError::UnsupportedFormat(other.into())),
    }
}

/// Process a library document: split into structural units, create parents,
/// and chunk each unit with token-based budgets.
///
/// Returns a `LibraryDocumentResult` containing all parent records and their
/// child chunks ready for embedding and storage.
pub fn process_library_document(
    file_path: &Path,
    source_format: &str,
    doc_id: &str,
    doc_fingerprint: &str,
    doc_title: Option<&str>,
    tokenizer: &ModelTokenizer,
    target_tokens: usize,
    overlap_tokens: usize,
) -> Result<LibraryDocumentResult, LibraryDocumentError> {
    let units = split_document(file_path, source_format)?;

    if units.is_empty() {
        return Ok(LibraryDocumentResult {
            units: vec![],
            metadata: HashMap::new(),
            total_chunks: 0,
        });
    }

    let mut processed_units = Vec::with_capacity(units.len());
    let mut total_chunks = 0;

    for unit in &units {
        // Create parent record
        let parent = create_parent_for_unit(doc_id, doc_fingerprint, unit);

        // Build header for indexed text
        let header = build_chunk_header(doc_title, &unit.title, &unit.unit_locator);

        // Chunk the unit text
        let token_chunks = tokenizer.chunk_by_tokens(&unit.text, target_tokens, overlap_tokens)?;

        let mut chunks = Vec::with_capacity(token_chunks.len());
        for (i, tc) in token_chunks.iter().enumerate() {
            let text_indexed = if header.is_empty() {
                tc.text.clone()
            } else {
                format!("{}\n{}", header, tc.text)
            };

            chunks.push(LibraryChunk {
                text_raw: tc.text.clone(),
                text_indexed,
                token_count: tc.token_count,
                char_start: tc.char_start,
                char_end: tc.char_end,
                chunk_index: i,
                parent_unit_id: parent.point_id.clone(),
            });
        }

        total_chunks += chunks.len();
        processed_units.push(ProcessedUnit { parent, chunks });
    }

    let mut metadata = HashMap::new();
    metadata.insert("unit_count".to_string(), processed_units.len().to_string());
    metadata.insert("total_chunks".to_string(), total_chunks.to_string());
    metadata.insert("source_format".to_string(), source_format.to_string());

    debug!(
        "Processed library document: {} units, {} total chunks",
        processed_units.len(),
        total_chunks
    );

    Ok(LibraryDocumentResult {
        units: processed_units,
        metadata,
        total_chunks,
    })
}

// ─── Internal helpers ────────────────────────────────────────────────────────

/// Create a parent record for a structural unit.
fn create_parent_for_unit(
    doc_id: &str,
    doc_fingerprint: &str,
    unit: &StructuralUnit,
) -> ParentUnitRecord {
    let point_id =
        parent_unit::parent_point_id(doc_id, &unit.unit_type, &unit.unit_locator);
    ParentUnitRecord {
        point_id,
        doc_id: doc_id.to_string(),
        doc_fingerprint: doc_fingerprint.to_string(),
        unit_type: unit.unit_type.clone(),
        unit_locator: unit.unit_locator.clone(),
        unit_text: unit.text.clone(),
        unit_char_len: unit.text.len(),
        unit_hash: parent_unit::sha256_hex(&unit.text),
    }
}

/// Build a header string for chunk indexing.
///
/// Format: `{doc_title} - {unit_label}` where unit_label is derived from
/// the unit's title or locator (e.g., "Page 5", "Slide 3", "Chapter: Introduction").
fn build_chunk_header(
    doc_title: Option<&str>,
    unit_title: &Option<String>,
    unit_locator: &serde_json::Value,
) -> String {
    let unit_label = if let Some(title) = unit_title {
        title.clone()
    } else if let Some(page) = unit_locator.get("page").and_then(|v| v.as_u64()) {
        format!("Page {}", page)
    } else if let Some(slide) = unit_locator.get("slide").and_then(|v| v.as_u64()) {
        format!("Slide {}", slide)
    } else if let Some(section) = unit_locator.get("section").and_then(|v| v.as_u64()) {
        format!("Section {}", section)
    } else if let Some(sheet) = unit_locator.get("sheet").and_then(|v| v.as_str()) {
        format!("Sheet: {}", sheet)
    } else if let Some(idx) = unit_locator.get("section_index").and_then(|v| v.as_u64()) {
        format!("Section {}", idx + 1)
    } else {
        return doc_title.unwrap_or("").to_string();
    };

    match doc_title {
        Some(title) if !title.is_empty() => format!("{} - {}", title, unit_label),
        _ => unit_label,
    }
}

/// Helper for splitting OpenDocument presentations/drawings by page elements.
fn split_opendocument_pages(
    file_path: &Path,
    format_name: &str,
    page_tag: &str,
    unit_type: &str,
    locator_key: &str,
) -> Result<Vec<StructuralUnit>, LibraryDocumentError> {
    let file = File::open(file_path)?;
    let mut archive = zip::ZipArchive::new(file).map_err(|e| LibraryDocumentError::Extraction {
        format: format_name.into(),
        message: e.to_string(),
    })?;

    let mut xml_content = String::new();
    if let Ok(mut content_file) = archive.by_name("content.xml") {
        content_file.read_to_string(&mut xml_content)?;
    } else {
        return Err(LibraryDocumentError::Extraction {
            format: format_name.into(),
            message: "No content.xml found".into(),
        });
    }

    let open_tag = format!("<{} ", page_tag);
    let pages: Vec<&str> = xml_content.split(&open_tag).collect();

    let mut units = Vec::new();
    for (i, page_xml) in pages.iter().skip(1).enumerate() {
        // Extract page name attribute
        let page_name =
            extract_xml_attr(page_xml, "draw:name").unwrap_or_else(|| format!("{}{}", locator_key, i + 1));

        let text = extract_text_from_xml_tags(page_xml, "text:p");
        let cleaned = clean_text(&text);
        if cleaned.is_empty() {
            continue;
        }
        units.push(StructuralUnit {
            unit_type: unit_type.to_string(),
            unit_locator: serde_json::json!({locator_key: i + 1}),
            text: cleaned,
            title: Some(page_name),
        });
    }

    debug!("{} split into {} pages", format_name.to_uppercase(), units.len());
    Ok(units)
}

/// Extract text from DOCX XML paragraphs (w:t tags within w:p elements).
fn extract_docx_text_from_xml(xml: &str) -> String {
    let mut text = String::new();
    let mut in_text_tag = false;
    let mut current_text = String::new();

    for part in xml.split('<') {
        if part.is_empty() {
            continue;
        }
        if part.starts_with("w:t") {
            in_text_tag = true;
            if let Some(content_start) = part.find('>') {
                current_text.push_str(&part[content_start + 1..]);
            }
        } else if part.starts_with("/w:t") {
            in_text_tag = false;
            if !current_text.is_empty() {
                text.push_str(&current_text);
                current_text.clear();
            }
        } else if part.starts_with("w:p") && !part.starts_with("w:pPr") {
            if !text.is_empty() && !text.ends_with('\n') {
                text.push('\n');
            }
        } else if in_text_tag {
            if let Some(end_pos) = part.find('>') {
                current_text.push_str(&part[end_pos + 1..]);
            } else {
                current_text.push_str(part);
            }
        }
    }

    text
}

/// Extract text content from XML tags (same algorithm as document_processor).
fn extract_text_from_xml_tags(xml_content: &str, tag_name: &str) -> String {
    let mut text = String::new();
    let mut in_tag = false;
    let mut depth = 0i32;

    for part in xml_content.split('<') {
        if part.is_empty() {
            continue;
        }

        let close_prefix = format!("/{}", tag_name);

        if part.starts_with(tag_name) {
            in_tag = true;
            depth += 1;
            if let Some(content_start) = part.find('>') {
                let content = &part[content_start + 1..];
                if !content.is_empty() {
                    text.push_str(content);
                }
            }
        } else if part.starts_with(&close_prefix) {
            depth -= 1;
            if depth <= 0 {
                in_tag = false;
                depth = 0;
                text.push('\n');
            }
        } else if in_tag {
            if let Some(pos) = part.find('>') {
                let content = &part[pos + 1..];
                if !content.is_empty() {
                    text.push_str(content);
                }
            }
        }
    }

    text
}

/// Extract the first XML attribute value from a tag fragment.
fn extract_xml_attr(xml_fragment: &str, attr_name: &str) -> Option<String> {
    let search = format!("{}=\"", attr_name);
    if let Some(start) = xml_fragment.find(&search) {
        let value_start = start + search.len();
        if let Some(end) = xml_fragment[value_start..].find('"') {
            return Some(xml_fragment[value_start..value_start + end].to_string());
        }
    }
    None
}

/// Extract first heading (h1-h3) from HTML content.
fn extract_html_heading(html: &str) -> Option<String> {
    for tag in &["h1", "h2", "h3"] {
        let open = format!("<{}", tag);
        if let Some(start) = html.to_lowercase().find(&open) {
            let rest = &html[start..];
            if let Some(gt) = rest.find('>') {
                let after_tag = &rest[gt + 1..];
                if let Some(close) = after_tag.find('<') {
                    let heading_text = after_tag[..close].trim();
                    if !heading_text.is_empty() {
                        return Some(heading_text.to_string());
                    }
                }
            }
        }
    }
    None
}

/// Strip RTF control codes, extracting plain text content.
fn strip_rtf_control_codes(rtf: &str) -> String {
    let mut text = String::new();
    let mut i = 0;
    let chars: Vec<char> = rtf.chars().collect();
    let mut brace_depth = 0i32;
    let mut skip_group = false;

    while i < chars.len() {
        match chars[i] {
            '{' => {
                brace_depth += 1;
                // Check if this is a special group to skip
                let rest: String = chars[i..].iter().take(20).collect();
                if rest.contains("\\fonttbl")
                    || rest.contains("\\colortbl")
                    || rest.contains("\\stylesheet")
                    || rest.contains("\\info")
                {
                    skip_group = true;
                }
                i += 1;
            }
            '}' => {
                brace_depth -= 1;
                if brace_depth <= 0 {
                    skip_group = false;
                }
                i += 1;
            }
            '\\' if !skip_group => {
                i += 1;
                if i >= chars.len() {
                    break;
                }
                match chars[i] {
                    '\n' | '\r' => {
                        i += 1;
                    }
                    '\'' => {
                        // Hex escape: \'XX
                        i += 1;
                        if i + 1 < chars.len() {
                            let hex: String = chars[i..i + 2].iter().collect();
                            if let Ok(byte) = u8::from_str_radix(&hex, 16) {
                                text.push(byte as char);
                            }
                            i += 2;
                        }
                    }
                    _ => {
                        // Control word — skip until space or non-alpha
                        let mut word = String::new();
                        while i < chars.len() && chars[i].is_ascii_alphabetic() {
                            word.push(chars[i]);
                            i += 1;
                        }
                        // Skip optional numeric parameter
                        while i < chars.len()
                            && (chars[i].is_ascii_digit() || chars[i] == '-')
                        {
                            i += 1;
                        }
                        // Skip trailing space delimiter
                        if i < chars.len() && chars[i] == ' ' {
                            i += 1;
                        }
                        // Translate known control words
                        match word.as_str() {
                            "par" | "line" => text.push('\n'),
                            "tab" => text.push('\t'),
                            _ => {}
                        }
                    }
                }
            }
            _ if !skip_group => {
                text.push(chars[i]);
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }

    text
}

/// Parse an ATX-style markdown heading (# through ######).
struct HeadingInfo {
    level: usize,
    title: String,
}

fn parse_atx_heading(line: &str) -> Option<HeadingInfo> {
    let trimmed = line.trim_start();
    if !trimmed.starts_with('#') {
        return None;
    }

    let level = trimmed.chars().take_while(|&c| c == '#').count();
    if level == 0 || level > 6 {
        return None;
    }

    let rest = &trimmed[level..];
    if !rest.is_empty() && !rest.starts_with(' ') && !rest.starts_with('\t') {
        return None; // No space after # — not a heading (e.g., #hashtag)
    }

    let title = rest.trim().trim_end_matches('#').trim().to_string();
    if title.is_empty() {
        return None;
    }

    Some(HeadingInfo { level, title })
}

/// Split text output into heading-based sections.
///
/// Looks for lines that appear to be headings: short, no trailing punctuation,
/// followed by content. Used for HTML-converted text.
fn split_text_by_headings(text: &str, unit_type: &str) -> Result<Vec<StructuralUnit>, LibraryDocumentError> {
    // For HTML-converted text, headings are typically rendered as lines
    // in ALL CAPS or with specific formatting. We use a simple heuristic:
    // lines that are short (<80 chars), non-empty, and followed by longer content.

    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return Ok(vec![]);
    }

    // Try markdown-style heading detection first
    let has_md_headings = lines.iter().any(|l| parse_atx_heading(l).is_some());
    if has_md_headings {
        return split_markdown_text(text);
    }

    // Fallback: treat entire text as single section
    let cleaned = clean_text(text);
    if cleaned.is_empty() {
        return Ok(vec![]);
    }

    Ok(vec![StructuralUnit {
        unit_type: unit_type.to_string(),
        unit_locator: serde_json::json!({"section_index": 0}),
        text: cleaned,
        title: None,
    }])
}

/// Clean extracted text: normalize whitespace, remove control characters.
fn clean_text(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut prev_was_whitespace = false;

    for ch in text.chars() {
        if ch == '\x0C' {
            continue; // Skip form feeds
        }
        if ch.is_control() && ch != '\n' && ch != '\t' {
            continue;
        }

        if ch.is_whitespace() {
            if !prev_was_whitespace || ch == '\n' {
                result.push(if ch == '\n' { '\n' } else { ' ' });
            }
            prev_was_whitespace = true;
        } else {
            result.push(ch);
            prev_was_whitespace = false;
        }
    }

    result.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Unit splitting tests ────────────────────────────────────────

    #[test]
    fn test_clean_text_basic() {
        assert_eq!(clean_text("  hello   world  "), "hello world");
        assert_eq!(clean_text("line1\n\nline2"), "line1\n\nline2");
        assert_eq!(clean_text(""), "");
    }

    #[test]
    fn test_clean_text_form_feeds() {
        assert_eq!(clean_text("page1\x0Cpage2"), "page1page2");
    }

    #[test]
    fn test_clean_text_control_chars() {
        assert_eq!(clean_text("hello\x01world"), "helloworld");
        // Tabs are whitespace, so they get normalized to spaces
        assert_eq!(clean_text("keep\ttabs"), "keep tabs");
        assert_eq!(clean_text("keep\nnewlines"), "keep\nnewlines");
    }

    // ─── ATX heading parsing ─────────────────────────────────────────

    #[test]
    fn test_parse_atx_heading_levels() {
        let h1 = parse_atx_heading("# Title").unwrap();
        assert_eq!(h1.level, 1);
        assert_eq!(h1.title, "Title");

        let h2 = parse_atx_heading("## Subtitle").unwrap();
        assert_eq!(h2.level, 2);
        assert_eq!(h2.title, "Subtitle");

        let h6 = parse_atx_heading("###### Deep").unwrap();
        assert_eq!(h6.level, 6);
        assert_eq!(h6.title, "Deep");
    }

    #[test]
    fn test_parse_atx_heading_not_heading() {
        assert!(parse_atx_heading("Not a heading").is_none());
        assert!(parse_atx_heading("#hashtag").is_none()); // No space after #
        assert!(parse_atx_heading("####### TooDeep").is_none()); // >6 levels
        assert!(parse_atx_heading("# ").is_none()); // Empty title
    }

    #[test]
    fn test_parse_atx_heading_trailing_hashes() {
        let h = parse_atx_heading("## Title ##").unwrap();
        assert_eq!(h.title, "Title");
    }

    // ─── Markdown splitting ──────────────────────────────────────────

    #[test]
    fn test_split_markdown_basic() {
        let md = "# Intro\n\nSome intro text.\n\n## Details\n\nMore details here.";
        let units = split_markdown_text(md).unwrap();
        assert_eq!(units.len(), 2);
        assert_eq!(units[0].title, Some("Intro".to_string()));
        assert!(units[0].text.contains("intro text"));
        assert_eq!(units[1].title, Some("Details".to_string()));
        assert!(units[1].text.contains("More details"));
    }

    #[test]
    fn test_split_markdown_frontmatter_skipped() {
        let md = "---\ntitle: Test\nauthor: Me\n---\n\n# Chapter 1\n\nContent.";
        let units = split_markdown_text(md).unwrap();
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].title, Some("Chapter 1".to_string()));
        assert!(!units[0].text.contains("title: Test"));
    }

    #[test]
    fn test_split_markdown_no_headings() {
        let md = "Just some plain text\nwith no headings at all.";
        let units = split_markdown_text(md).unwrap();
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].unit_type, UNIT_TYPE_TEXT_SECTION);
    }

    // ─── Plain text splitting ────────────────────────────────────────

    #[test]
    fn test_split_text_by_paragraphs_single() {
        let text = "Short text.";
        let units = split_text_by_paragraphs(text).unwrap();
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].text, "Short text.");
    }

    #[test]
    fn test_split_text_by_paragraphs_multiple() {
        // Create text that exceeds the 2000 char threshold per section
        let long_para = "A".repeat(1500);
        let text = format!("{}\n\n{}\n\n{}", long_para, long_para, long_para);
        let units = split_text_by_paragraphs(&text).unwrap();
        assert!(units.len() >= 2, "Long text should split into multiple sections");
    }

    #[test]
    fn test_split_text_empty() {
        let units = split_text_by_paragraphs("").unwrap();
        assert!(units.is_empty());
    }

    // ─── XML helpers ─────────────────────────────────────────────────

    #[test]
    fn test_extract_xml_attr() {
        assert_eq!(
            extract_xml_attr(r#"table:name="Sheet1" table:style-name="ta1""#, "table:name"),
            Some("Sheet1".to_string())
        );
        assert_eq!(
            extract_xml_attr(r#"draw:name="Slide 1""#, "draw:name"),
            Some("Slide 1".to_string())
        );
        assert_eq!(extract_xml_attr("no-attr-here", "table:name"), None);
    }

    #[test]
    fn test_extract_text_from_xml_tags() {
        let xml = r#"<text:p text:style-name="P1">Hello world</text:p>"#;
        let text = extract_text_from_xml_tags(xml, "text:p");
        assert!(text.contains("Hello world"));
    }

    #[test]
    fn test_extract_docx_text_from_xml() {
        let xml = r#"<w:p><w:r><w:t>Hello</w:t></w:r></w:p><w:p><w:r><w:t>World</w:t></w:r></w:p>"#;
        let text = extract_docx_text_from_xml(xml);
        assert!(text.contains("Hello"));
        assert!(text.contains("World"));
    }

    // ─── RTF stripping ───────────────────────────────────────────────

    #[test]
    fn test_strip_rtf_basic() {
        let rtf = r"{\rtf1\ansi Hello World}";
        let text = strip_rtf_control_codes(rtf);
        assert!(text.contains("Hello World"), "Got: {}", text);
    }

    #[test]
    fn test_strip_rtf_paragraphs() {
        let rtf = r"{\rtf1 Line one\par Line two}";
        let text = strip_rtf_control_codes(rtf);
        assert!(text.contains("Line one"));
        assert!(text.contains("Line two"));
    }

    // ─── HTML heading extraction ─────────────────────────────────────

    #[test]
    fn test_extract_html_heading() {
        assert_eq!(
            extract_html_heading("<h1>Title Here</h1><p>Content</p>"),
            Some("Title Here".to_string())
        );
        assert_eq!(
            extract_html_heading("<p>No heading</p>"),
            None
        );
        assert_eq!(
            extract_html_heading("<H2>Mixed Case</H2>"),
            Some("Mixed Case".to_string())
        );
    }

    // ─── Header building ─────────────────────────────────────────────

    #[test]
    fn test_build_chunk_header_page() {
        let locator = serde_json::json!({"page": 5});
        let header = build_chunk_header(Some("My PDF"), &None, &locator);
        assert_eq!(header, "My PDF - Page 5");
    }

    #[test]
    fn test_build_chunk_header_slide() {
        let locator = serde_json::json!({"slide": 3});
        let header = build_chunk_header(Some("Presentation"), &None, &locator);
        assert_eq!(header, "Presentation - Slide 3");
    }

    #[test]
    fn test_build_chunk_header_with_unit_title() {
        let locator = serde_json::json!({"spine_id": "ch1"});
        let title = Some("Introduction".to_string());
        let header = build_chunk_header(Some("My Book"), &title, &locator);
        assert_eq!(header, "My Book - Introduction");
    }

    #[test]
    fn test_build_chunk_header_no_doc_title() {
        let locator = serde_json::json!({"page": 1});
        let header = build_chunk_header(None, &None, &locator);
        assert_eq!(header, "Page 1");
    }

    #[test]
    fn test_build_chunk_header_sheet() {
        let locator = serde_json::json!({"sheet": "Revenue"});
        let header = build_chunk_header(Some("Budget"), &None, &locator);
        assert_eq!(header, "Budget - Sheet: Revenue");
    }

    // ─── Process integration (requires tokenizer) ────────────────────

    fn get_test_tokenizer() -> Option<ModelTokenizer> {
        ModelTokenizer::from_model_cache(None).ok()
    }

    #[test]
    fn test_process_markdown_document() {
        let tokenizer = match get_test_tokenizer() {
            Some(t) => t,
            None => return, // Skip if model not cached
        };

        // Write a temporary markdown file
        let dir = std::env::temp_dir().join("wqm_test_lib_doc");
        let _ = std::fs::create_dir_all(&dir);
        let md_path = dir.join("test.md");
        std::fs::write(
            &md_path,
            "# Chapter 1\n\nThis is the first chapter with enough text to be meaningful.\n\n## Section 1.1\n\nDetailed content goes here with more words to fill it out.",
        )
        .unwrap();

        let result = process_library_document(
            &md_path,
            "markdown",
            "doc-test-1",
            "fp-test-1",
            Some("Test Document"),
            &tokenizer,
            105,
            12,
        )
        .unwrap();

        assert!(!result.units.is_empty(), "Should have at least one unit");
        assert!(result.total_chunks > 0, "Should have at least one chunk");

        // Verify parent-child relationships
        for unit in &result.units {
            assert_eq!(unit.parent.doc_id, "doc-test-1");
            for chunk in &unit.chunks {
                assert_eq!(chunk.parent_unit_id, unit.parent.point_id);
                assert!(!chunk.text_raw.is_empty());
                assert!(chunk.text_indexed.contains("Test Document"));
            }
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_process_plain_text_document() {
        let tokenizer = match get_test_tokenizer() {
            Some(t) => t,
            None => return,
        };

        let dir = std::env::temp_dir().join("wqm_test_lib_doc_txt");
        let _ = std::fs::create_dir_all(&dir);
        let txt_path = dir.join("test.txt");
        std::fs::write(&txt_path, "A simple paragraph of text for testing purposes.").unwrap();

        let result = process_library_document(
            &txt_path,
            "text",
            "doc-txt-1",
            "fp-txt-1",
            None,
            &tokenizer,
            105,
            12,
        )
        .unwrap();

        assert_eq!(result.units.len(), 1);
        assert!(result.total_chunks >= 1);
        // Without doc title, header should still work
        let chunk = &result.units[0].chunks[0];
        assert!(!chunk.text_raw.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_split_document_unsupported_format() {
        let result = split_document(Path::new("test.xyz"), "xyz");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Unsupported format"));
    }

    #[test]
    fn test_structural_unit_types() {
        // Verify unit type constants are consistent
        assert_eq!(UNIT_TYPE_PDF_PAGE, "pdf_page");
        assert_eq!(UNIT_TYPE_EPUB_SECTION, "epub_section");
        assert_eq!(UNIT_TYPE_DOCX_SECTION, "docx_section");
        assert_eq!(UNIT_TYPE_TEXT_SECTION, "text_section");
    }
}
