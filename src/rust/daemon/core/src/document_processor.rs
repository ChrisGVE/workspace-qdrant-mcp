//! Document Processing Module
//!
//! Task 437: Provides file parsing and text extraction for various document formats.
//!
//! Supported file types:
//! - Code files (UTF-8 text with language-specific metadata extraction)
//! - PDF documents (using pdf-extract crate)
//! - EPUB documents (using epub crate)
//! - DOCX documents (using zip crate to extract XML content)
//! - Plain text, Markdown, HTML, XML, JSON
//!
//! The DocumentProcessor integrates with the queue_processor to extract text content
//! before Qdrant ingestion. For code files, it can optionally use LSP for enhanced
//! symbol extraction.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;
use thiserror::Error;
use tracing::{debug, info, warn};

use crate::{ChunkingConfig, DocumentContent, DocumentType, TextChunk};
use crate::file_classification::{classify_file_type, FileType};

/// Document processor errors
#[derive(Error, Debug)]
pub enum DocumentProcessorError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("File not found: {0}")]
    FileNotFound(String),

    #[error("Unsupported file format: {0}")]
    UnsupportedFormat(String),

    #[error("Encoding error: {0}")]
    EncodingError(String),

    #[error("PDF extraction error: {0}")]
    PdfError(String),

    #[error("EPUB extraction error: {0}")]
    EpubError(String),

    #[error("DOCX extraction error: {0}")]
    DocxError(String),

    #[error("Parse error: {0}")]
    ParseError(String),
}

/// Result type for document processor operations
pub type DocumentProcessorResult<T> = Result<T, DocumentProcessorError>;

/// Processed document with extracted content and metadata
#[derive(Debug, Clone)]
pub struct ProcessedDocument {
    /// Extracted text content
    pub content: String,
    /// Document metadata
    pub metadata: HashMap<String, String>,
    /// Detected file type
    pub file_type: FileType,
    /// Detected language (for code files)
    pub language: Option<String>,
}

/// Document processor for extracting text from various file formats
pub struct DocumentProcessor {
    /// Chunking configuration
    chunking_config: ChunkingConfig,
}

impl DocumentProcessor {
    /// Create a new document processor with default chunking configuration
    pub fn new() -> Self {
        Self {
            chunking_config: ChunkingConfig::default(),
        }
    }

    /// Create a new document processor with custom chunking configuration
    pub fn with_config(chunking_config: ChunkingConfig) -> Self {
        Self { chunking_config }
    }

    /// Process a file and extract text content
    ///
    /// This is the main entry point for document processing. It:
    /// 1. Detects the file type
    /// 2. Extracts text content using the appropriate parser
    /// 3. Generates metadata about the document
    /// 4. Returns a ProcessedDocument
    pub async fn process_file(&self, path: &Path) -> DocumentProcessorResult<ProcessedDocument> {
        if !path.exists() {
            return Err(DocumentProcessorError::FileNotFound(
                path.display().to_string(),
            ));
        }

        let file_type = classify_file_type(path);
        let extension = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        info!("Processing file: {} (type: {:?}, ext: {})", path.display(), file_type, extension);

        let (content, language) = match extension.as_str() {
            // PDF files
            "pdf" => (self.extract_pdf(path).await?, None),

            // EPUB files
            "epub" => (self.extract_epub(path)?, None),

            // DOCX files
            "docx" => (self.extract_docx(path)?, None),

            // Code files - extract with language detection
            "py" => (self.extract_text_file(path).await?, Some("python".to_string())),
            "rs" => (self.extract_text_file(path).await?, Some("rust".to_string())),
            "js" => (self.extract_text_file(path).await?, Some("javascript".to_string())),
            "ts" => (self.extract_text_file(path).await?, Some("typescript".to_string())),
            "java" => (self.extract_text_file(path).await?, Some("java".to_string())),
            "go" => (self.extract_text_file(path).await?, Some("go".to_string())),
            "rb" => (self.extract_text_file(path).await?, Some("ruby".to_string())),
            "php" => (self.extract_text_file(path).await?, Some("php".to_string())),
            "c" | "h" => (self.extract_text_file(path).await?, Some("c".to_string())),
            "cpp" | "hpp" | "cc" | "cxx" => (self.extract_text_file(path).await?, Some("cpp".to_string())),
            "cs" => (self.extract_text_file(path).await?, Some("csharp".to_string())),
            "swift" => (self.extract_text_file(path).await?, Some("swift".to_string())),
            "kt" | "kts" => (self.extract_text_file(path).await?, Some("kotlin".to_string())),
            "scala" => (self.extract_text_file(path).await?, Some("scala".to_string())),
            "lua" => (self.extract_text_file(path).await?, Some("lua".to_string())),
            "sh" | "bash" => (self.extract_text_file(path).await?, Some("shell".to_string())),

            // Markup and data files
            "md" | "markdown" => (self.extract_text_file(path).await?, Some("markdown".to_string())),
            "html" | "htm" => (self.extract_text_file(path).await?, Some("html".to_string())),
            "xml" => (self.extract_text_file(path).await?, Some("xml".to_string())),
            "json" => (self.extract_text_file(path).await?, Some("json".to_string())),
            "yaml" | "yml" => (self.extract_text_file(path).await?, Some("yaml".to_string())),
            "toml" => (self.extract_text_file(path).await?, Some("toml".to_string())),

            // Plain text files
            "txt" | "text" | "log" => (self.extract_text_file(path).await?, None),

            // Default: try to read as text
            _ => {
                match self.extract_text_file(path).await {
                    Ok(content) => (content, None),
                    Err(_) => {
                        return Err(DocumentProcessorError::UnsupportedFormat(
                            format!("Unable to extract text from: {}", extension),
                        ));
                    }
                }
            }
        };

        // Build metadata
        let mut metadata = HashMap::new();
        metadata.insert("file_path".to_string(), path.display().to_string());
        metadata.insert("file_name".to_string(), path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string());
        metadata.insert("extension".to_string(), extension.clone());
        metadata.insert("content_length".to_string(), content.len().to_string());
        if let Some(ref lang) = language {
            metadata.insert("language".to_string(), lang.clone());
        }

        Ok(ProcessedDocument {
            content,
            metadata,
            file_type,
            language,
        })
    }

    /// Extract document content with chunking for queue_processor integration
    ///
    /// This method is called by queue_processor to get chunked content
    /// ready for embedding and storage.
    pub async fn extract_document_content(&self, path: &Path) -> DocumentProcessorResult<DocumentContent> {
        let processed = self.process_file(path).await?;

        // Convert FileType to DocumentType
        let document_type = self.file_type_to_document_type(&processed.file_type, &processed.language);

        // Generate chunks
        let chunks = self.chunk_text(&processed.content);

        Ok(DocumentContent {
            raw_text: processed.content,
            metadata: processed.metadata,
            document_type,
            chunks,
        })
    }

    /// Extract text from a UTF-8 text file
    async fn extract_text_file(&self, path: &Path) -> DocumentProcessorResult<String> {
        // Try UTF-8 first
        match tokio::fs::read_to_string(path).await {
            Ok(content) => {
                debug!("Successfully read {} as UTF-8", path.display());
                Ok(content)
            }
            Err(_) => {
                // Try with encoding detection
                let bytes = tokio::fs::read(path).await?;
                self.decode_with_detection(&bytes, path)
            }
        }
    }

    /// Decode bytes with encoding detection fallback
    fn decode_with_detection(&self, bytes: &[u8], path: &Path) -> DocumentProcessorResult<String> {
        // Try chardet for encoding detection
        let result = chardet::detect(bytes);
        let encoding_name = result.0.to_lowercase();

        debug!("Detected encoding: {} (confidence: {}) for {}", encoding_name, result.1, path.display());

        // Use encoding_rs for decoding
        let encoding = encoding_rs::Encoding::for_label(encoding_name.as_bytes())
            .unwrap_or(encoding_rs::UTF_8);

        let (decoded, _, had_errors) = encoding.decode(bytes);
        if had_errors {
            warn!("Encoding errors while decoding {}", path.display());
        }

        Ok(decoded.into_owned())
    }

    /// Extract text from a PDF file
    async fn extract_pdf(&self, path: &Path) -> DocumentProcessorResult<String> {
        let path_owned = path.to_path_buf();

        // Run PDF extraction in a blocking thread
        tokio::task::spawn_blocking(move || {
            pdf_extract::extract_text(&path_owned)
                .map_err(|e| DocumentProcessorError::PdfError(e.to_string()))
        })
        .await
        .map_err(|e| DocumentProcessorError::PdfError(format!("Task join error: {}", e)))?
    }

    /// Extract text from an EPUB file
    fn extract_epub(&self, path: &Path) -> DocumentProcessorResult<String> {
        let doc = epub::doc::EpubDoc::new(path)
            .map_err(|e| DocumentProcessorError::EpubError(e.to_string()))?;

        let mut content = String::new();
        let mut doc = doc; // Make mutable

        // Get spine (ordered list of content documents)
        // SpineItem contains idref field with the item ID
        let spine_ids: Vec<String> = doc.spine.iter()
            .map(|item| item.idref.clone())
            .collect();

        for item_id in spine_ids {
            if let Some((data, _mime)) = doc.get_resource(&item_id) {
                // Parse HTML content
                let text = self.extract_text_from_html(&data)?;
                if !text.is_empty() {
                    if !content.is_empty() {
                        content.push_str("\n\n");
                    }
                    content.push_str(&text);
                }
            }
        }

        Ok(content)
    }

    /// Extract text from HTML content (used by EPUB parser)
    fn extract_text_from_html(&self, data: &[u8]) -> DocumentProcessorResult<String> {
        let html = String::from_utf8_lossy(data);

        // Simple HTML tag stripping (basic approach)
        // For production, consider using html5ever or scraper crate
        let mut text = String::new();
        let mut in_tag = false;
        let mut in_script = false;
        let mut in_style = false;

        let lowercase_html = html.to_lowercase();

        for (i, c) in html.chars().enumerate() {
            if c == '<' {
                in_tag = true;
                // Check for script/style tags
                if lowercase_html[i..].starts_with("<script") {
                    in_script = true;
                } else if lowercase_html[i..].starts_with("<style") {
                    in_style = true;
                }
            } else if c == '>' {
                in_tag = false;
                // Check for closing script/style tags
                if lowercase_html[..=i].ends_with("</script>") {
                    in_script = false;
                } else if lowercase_html[..=i].ends_with("</style>") {
                    in_style = false;
                }
            } else if !in_tag && !in_script && !in_style {
                text.push(c);
            }
        }

        // Decode HTML entities
        let text = text
            .replace("&nbsp;", " ")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&amp;", "&")
            .replace("&quot;", "\"")
            .replace("&#39;", "'");

        // Normalize whitespace
        let text = text.split_whitespace().collect::<Vec<_>>().join(" ");

        Ok(text)
    }

    /// Extract text from a DOCX file
    fn extract_docx(&self, path: &Path) -> DocumentProcessorResult<String> {
        let file = std::fs::File::open(path)?;
        let mut archive = zip::ZipArchive::new(file)
            .map_err(|e| DocumentProcessorError::DocxError(e.to_string()))?;

        // Read the main document content
        let mut document_xml = String::new();
        if let Ok(mut file) = archive.by_name("word/document.xml") {
            file.read_to_string(&mut document_xml)?;
        } else {
            return Err(DocumentProcessorError::DocxError(
                "word/document.xml not found in DOCX".to_string(),
            ));
        }

        // Extract text from XML
        self.extract_text_from_docx_xml(&document_xml)
    }

    /// Extract text from DOCX XML content
    fn extract_text_from_docx_xml(&self, xml: &str) -> DocumentProcessorResult<String> {
        let mut text = String::new();

        // Find all text runs (<w:t>content</w:t>)
        let mut i = 0;
        let bytes = xml.as_bytes();

        while i < bytes.len() {
            // Look for <w:t> or <w:t ...>
            if i + 5 < bytes.len() && &bytes[i..i+4] == b"<w:t" {
                // Find the closing >
                if let Some(start_pos) = xml[i..].find('>') {
                    let content_start = i + start_pos + 1;
                    // Find the closing </w:t>
                    if let Some(end_pos) = xml[content_start..].find("</w:t>") {
                        let content = &xml[content_start..content_start + end_pos];
                        text.push_str(content);
                        i = content_start + end_pos + 6;
                        continue;
                    }
                }
            }

            // Look for paragraph breaks <w:p> to add newlines
            if i + 5 < bytes.len() && &bytes[i..i+5] == b"</w:p" {
                if !text.ends_with('\n') {
                    text.push('\n');
                }
            }

            i += 1;
        }

        Ok(text.trim().to_string())
    }

    /// Convert FileType to DocumentType
    fn file_type_to_document_type(&self, file_type: &FileType, language: &Option<String>) -> DocumentType {
        match file_type {
            FileType::Code => {
                if let Some(lang) = language {
                    DocumentType::Code(lang.clone())
                } else {
                    DocumentType::Code("unknown".to_string())
                }
            }
            FileType::Test => {
                if let Some(lang) = language {
                    DocumentType::Code(lang.clone())
                } else {
                    DocumentType::Code("test".to_string())
                }
            }
            FileType::Docs => DocumentType::Markdown, // Most docs are markdown
            FileType::Config => DocumentType::Text,
            FileType::Data => DocumentType::Text,
            FileType::Build => DocumentType::Text,
            FileType::Other => DocumentType::Text,
        }
    }

    /// Chunk text content into smaller pieces for embedding
    fn chunk_text(&self, text: &str) -> Vec<TextChunk> {
        let mut chunks = Vec::new();
        let total_chars = text.len();

        if total_chars == 0 {
            return chunks;
        }

        let config = &self.chunking_config;

        if config.preserve_paragraphs {
            // Split by paragraphs first, then chunk within paragraphs
            self.chunk_by_paragraphs(text, &mut chunks);
        } else {
            // Simple character-based chunking
            self.chunk_by_size(text, &mut chunks);
        }

        chunks
    }

    /// Chunk text by paragraphs, merging small paragraphs
    fn chunk_by_paragraphs(&self, text: &str, chunks: &mut Vec<TextChunk>) {
        let config = &self.chunking_config;
        let paragraphs: Vec<&str> = text.split("\n\n").collect();

        let mut current_chunk = String::new();
        let mut current_start = 0;
        let mut chunk_index = 0;

        for para in paragraphs {
            let para = para.trim();
            if para.is_empty() {
                continue;
            }

            // If adding this paragraph would exceed chunk size
            if !current_chunk.is_empty()
                && current_chunk.len() + para.len() + 2 > config.chunk_size
            {
                // Save current chunk
                chunks.push(TextChunk {
                    content: current_chunk.clone(),
                    chunk_index,
                    start_char: current_start,
                    end_char: current_start + current_chunk.len(),
                    metadata: HashMap::new(),
                });
                chunk_index += 1;
                current_start += current_chunk.len();
                current_chunk.clear();
            }

            // If single paragraph is larger than chunk size, split it
            if para.len() > config.chunk_size {
                if !current_chunk.is_empty() {
                    chunks.push(TextChunk {
                        content: current_chunk.clone(),
                        chunk_index,
                        start_char: current_start,
                        end_char: current_start + current_chunk.len(),
                        metadata: HashMap::new(),
                    });
                    chunk_index += 1;
                    current_start += current_chunk.len();
                    current_chunk.clear();
                }

                // Split large paragraph
                self.chunk_large_paragraph(para, &mut chunk_index, &mut current_start, chunks);
            } else {
                // Add paragraph to current chunk
                if !current_chunk.is_empty() {
                    current_chunk.push_str("\n\n");
                }
                current_chunk.push_str(para);
            }
        }

        // Don't forget the last chunk
        if !current_chunk.is_empty() {
            chunks.push(TextChunk {
                content: current_chunk.clone(),
                chunk_index,
                start_char: current_start,
                end_char: current_start + current_chunk.len(),
                metadata: HashMap::new(),
            });
        }
    }

    /// Chunk a large paragraph that exceeds the chunk size
    fn chunk_large_paragraph(
        &self,
        text: &str,
        chunk_index: &mut usize,
        current_start: &mut usize,
        chunks: &mut Vec<TextChunk>,
    ) {
        let config = &self.chunking_config;
        let words: Vec<&str> = text.split_whitespace().collect();

        let mut current_chunk = String::new();

        for word in words {
            if current_chunk.len() + word.len() + 1 > config.chunk_size {
                if !current_chunk.is_empty() {
                    chunks.push(TextChunk {
                        content: current_chunk.clone(),
                        chunk_index: *chunk_index,
                        start_char: *current_start,
                        end_char: *current_start + current_chunk.len(),
                        metadata: HashMap::new(),
                    });
                    *chunk_index += 1;
                    *current_start += current_chunk.len();
                    current_chunk.clear();
                }
            }

            if !current_chunk.is_empty() {
                current_chunk.push(' ');
            }
            current_chunk.push_str(word);
        }

        if !current_chunk.is_empty() {
            chunks.push(TextChunk {
                content: current_chunk.clone(),
                chunk_index: *chunk_index,
                start_char: *current_start,
                end_char: *current_start + current_chunk.len(),
                metadata: HashMap::new(),
            });
            *chunk_index += 1;
            *current_start += current_chunk.len();
        }
    }

    /// Simple size-based chunking with overlap
    fn chunk_by_size(&self, text: &str, chunks: &mut Vec<TextChunk>) {
        let config = &self.chunking_config;
        let total_chars = text.len();

        let mut start = 0;
        let mut chunk_index = 0;

        while start < total_chars {
            let end = (start + config.chunk_size).min(total_chars);

            // Try to end at a word boundary
            let chunk_end = if end < total_chars {
                text[start..end]
                    .rfind(char::is_whitespace)
                    .map(|pos| start + pos + 1) // Include the whitespace, move past it
                    .unwrap_or(end)
            } else {
                end
            };

            // Ensure we make progress (avoid infinite loop)
            let chunk_end = if chunk_end <= start {
                end // Fall back to hard boundary if no whitespace found
            } else {
                chunk_end
            };

            let chunk_text = text[start..chunk_end].trim();

            if !chunk_text.is_empty() {
                chunks.push(TextChunk {
                    content: chunk_text.to_string(),
                    chunk_index,
                    start_char: start,
                    end_char: chunk_end,
                    metadata: HashMap::new(),
                });
                chunk_index += 1;
            }

            // Move start position with overlap, ensuring forward progress
            let next_start = if chunk_end > config.overlap_size {
                chunk_end - config.overlap_size
            } else {
                chunk_end
            };

            // Ensure we always move forward
            start = if next_start <= start {
                chunk_end
            } else {
                next_start
            };
        }
    }
}

impl Default for DocumentProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use std::fs::File;
    use std::io::Write;

    #[tokio::test]
    async fn test_process_text_file() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("readme.txt");
        let mut file = File::create(&file_path).unwrap();
        writeln!(file, "Hello, world!\nThis is a sample file.").unwrap();

        let processor = DocumentProcessor::new();
        let result = processor.process_file(&file_path).await.unwrap();

        assert!(result.content.contains("Hello, world!"));
        assert!(result.content.contains("sample file"));
        // .txt files are classified as Docs (documentation)
        assert_eq!(result.file_type, FileType::Docs);
    }

    #[tokio::test]
    async fn test_process_python_file() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("main.py");
        let mut file = File::create(&file_path).unwrap();
        writeln!(file, "def hello():\n    print('Hello, world!')").unwrap();

        let processor = DocumentProcessor::new();
        let result = processor.process_file(&file_path).await.unwrap();

        assert!(result.content.contains("def hello"));
        assert_eq!(result.language, Some("python".to_string()));
        // File classification depends on path/name - tempdir paths vary
        // Key assertion is that language detection works correctly
        assert!(matches!(result.file_type, FileType::Code | FileType::Test));
    }

    #[tokio::test]
    async fn test_extract_document_content() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("doc.txt");
        let mut file = File::create(&file_path).unwrap();

        // Create content larger than default chunk size
        let content = "This is paragraph one. ".repeat(50)
            + "\n\n"
            + &"This is paragraph two. ".repeat(50);
        write!(file, "{}", content).unwrap();

        let processor = DocumentProcessor::new();
        let result = processor.extract_document_content(&file_path).await.unwrap();

        assert!(!result.chunks.is_empty());
        assert!(result.chunks.len() > 1); // Should have multiple chunks
    }

    #[tokio::test]
    async fn test_file_not_found() {
        let processor = DocumentProcessor::new();
        let result = processor.process_file(Path::new("/nonexistent/file.txt")).await;

        assert!(matches!(result, Err(DocumentProcessorError::FileNotFound(_))));
    }

    #[test]
    fn test_chunk_text() {
        let processor = DocumentProcessor::with_config(ChunkingConfig {
            chunk_size: 100,
            overlap_size: 10,
            preserve_paragraphs: false,
        });

        let text = "Hello ".repeat(50);
        let chunks = processor.chunk_text(&text);

        assert!(!chunks.is_empty());
        // Each chunk should be <= chunk_size
        for chunk in &chunks {
            assert!(chunk.content.len() <= 100);
        }
    }

    #[test]
    fn test_extract_text_from_html() {
        let processor = DocumentProcessor::new();
        let html = b"<html><body><p>Hello</p><script>var x = 1;</script><p>World</p></body></html>";

        let result = processor.extract_text_from_html(html).unwrap();
        assert!(result.contains("Hello"));
        assert!(result.contains("World"));
        assert!(!result.contains("var x")); // Script should be stripped
    }
}
