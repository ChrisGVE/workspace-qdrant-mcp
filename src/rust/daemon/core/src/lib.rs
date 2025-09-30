//! Core processing engine for workspace-qdrant-mcp
//!
//! This crate provides the core document processing, file watching, and embedding
//! generation capabilities for the workspace-qdrant-mcp ingestion engine.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::HashMap;
use thiserror::Error;
use tokio::sync::Mutex;

pub mod config;
pub mod embedding;
pub mod error;
pub mod ipc;
pub mod logging;
pub mod daemon_state;
pub mod patterns;
pub mod processing;
pub mod queue_config;
pub mod queue_operations;
pub mod queue_error_handler;
pub mod service_discovery;
pub mod storage;
pub mod unified_config;
// Temporarily disable watching module for compilation
// pub mod watching;

use crate::processing::{Pipeline, TaskSubmitter, TaskSource, TaskPayload, TaskResult};
use crate::ipc::{IpcServer, IpcClient};
use crate::storage::StorageClient;
use crate::config::{Config, DaemonConfig};
use crate::unified_config::{UnifiedConfigManager, UnifiedConfigError, ConfigFormat};
pub use crate::embedding::{
    EmbeddingGenerator, EmbeddingConfig, EmbeddingResult,
    DenseEmbedding, SparseEmbedding, EmbeddingError
};
pub use crate::processing::{
    TaskPriority
};
pub use crate::error::{
    WorkspaceError, ErrorSeverity, ErrorRecoveryStrategy, 
    CircuitBreaker, ErrorMonitor, ErrorRecovery, Result
};
pub use crate::logging::{
    LoggingConfig, PerformanceMetrics, initialize_logging, initialize_daemon_silence,
    track_async_operation, log_error_with_context, LoggingErrorMonitor
};
pub use crate::daemon_state::{
    DaemonStateManager
};
pub use crate::service_discovery::{
    DiscoveryManager, ServiceRegistry, ServiceInfo, ServiceStatus,
    NetworkDiscovery, DiscoveryMessage, DiscoveryMessageType,
    HealthChecker, HealthStatus, HealthConfig,
    DiscoveryConfig, DiscoveryError, DiscoveryResult
};
pub use crate::patterns::{
    PatternManager, PatternError, PatternResult, AllPatterns,
    ProjectIndicators, ExcludePatterns, IncludePatterns, LanguageExtensions,
    Ecosystem, ProjectIndicator, ConfidenceLevel, LanguageGroup, CommentSyntax
};

/// Core processing errors
#[derive(Error, Debug)]
pub enum ProcessingError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parsing error: {0}")]
    Parse(String),

    #[error("Processing error: {0}")]
    Processing(String),

    #[error("Storage error: {0}")]
    Storage(String),
}

/// Document processing result
#[derive(Debug, Clone)]
pub struct DocumentResult {
    pub document_id: String,
    pub collection: String,
    pub chunks_created: Option<usize>,
    pub processing_time_ms: u64,
}

/// Processing statistics for monitoring
#[derive(Debug, Clone)]
pub struct ProcessingStats {
    pub total_documents_processed: u64,
    pub total_chunks_created: u64,
    pub average_processing_time_ms: u64,
    pub chunking_config: ChunkingConfig,
}

/// Document type enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum DocumentType {
    Pdf,
    Epub,
    Docx,
    Text,
    Markdown,
    Code(String), // Language name
    Unknown,
}

/// Text chunk with metadata
#[derive(Debug, Clone)]
pub struct TextChunk {
    pub content: String,
    pub chunk_index: usize,
    pub start_char: usize,
    pub end_char: usize,
    pub metadata: HashMap<String, String>,
}

/// Document content representation
#[derive(Debug, Clone)]
pub struct DocumentContent {
    pub raw_text: String,
    pub metadata: HashMap<String, String>,
    pub document_type: DocumentType,
    pub chunks: Vec<TextChunk>,
}

/// Chunking configuration
#[derive(Debug, Clone)]
pub struct ChunkingConfig {
    pub chunk_size: usize,
    pub overlap_size: usize,
    pub preserve_paragraphs: bool,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            overlap_size: 50,
            preserve_paragraphs: true,
        }
    }
}

/// Comprehensive document processor with format-specific parsing
#[derive(Clone)]
pub struct DocumentProcessor {
    chunking_config: ChunkingConfig,
}

impl DocumentProcessor {
    pub fn new() -> Self {
        Self {
            chunking_config: ChunkingConfig::default(),
        }
    }

    pub fn with_chunking_config(chunking_config: ChunkingConfig) -> Self {
        Self {
            chunking_config,
        }
    }

    pub async fn process_file(
        &self,
        file_path: &Path,
        collection: &str,
    ) -> std::result::Result<DocumentResult, ProcessingError> {
        let start_time = Instant::now();
        
        // Extract document content based on file type
        let document_content = self.extract_document_content(file_path).await?;
        
        let chunks_created = document_content.chunks.len();
        let document_id = uuid::Uuid::new_v4().to_string();
        
        // TODO: Store chunks in Qdrant (will be implemented in later tasks)
        tracing::info!(
            "Processed document '{}': {} chunks created, type: {:?}",
            file_path.display(),
            chunks_created,
            document_content.document_type
        );
        
        let processing_time_ms = start_time.elapsed().as_millis() as u64;
        
        Ok(DocumentResult {
            document_id,
            collection: collection.to_string(),
            chunks_created: Some(chunks_created),
            processing_time_ms,
        })
    }

    async fn extract_document_content(&self, file_path: &Path) -> std::result::Result<DocumentContent, ProcessingError> {
        // Detect file type using MIME type and extension
        let document_type = self.detect_document_type(file_path)?;
        
        // Extract raw text based on document type
        let raw_text = match &document_type {
            DocumentType::Pdf => self.extract_pdf_text(file_path).await?,
            DocumentType::Epub => self.extract_epub_text(file_path).await?,
            DocumentType::Docx => self.extract_docx_text(file_path).await?,
            DocumentType::Text | DocumentType::Markdown => {
                self.extract_text_file_content(file_path).await?
            },
            DocumentType::Code(_) => self.extract_code_file_content(file_path).await?,
            DocumentType::Unknown => {
                // Try to read as text with encoding detection
                self.extract_text_file_content(file_path).await?
            }
        };
        
        // Create chunks from the extracted text
        let chunks = self.create_text_chunks(&raw_text, &document_type)?;
        
        // Build metadata
        let mut metadata = HashMap::new();
        metadata.insert("file_path".to_string(), file_path.to_string_lossy().to_string());
        metadata.insert("file_name".to_string(), 
            file_path.file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "unknown".to_string())
        );
        metadata.insert("document_type".to_string(), format!("{:?}", document_type));
        metadata.insert("char_count".to_string(), raw_text.len().to_string());
        metadata.insert("chunk_count".to_string(), chunks.len().to_string());
        
        if let Ok(file_metadata) = tokio::fs::metadata(file_path).await {
            if let Ok(modified) = file_metadata.modified() {
                metadata.insert("last_modified".to_string(), 
                    chrono::DateTime::<chrono::Utc>::from(modified).to_rfc3339());
            }
            metadata.insert("file_size".to_string(), file_metadata.len().to_string());
        }
        
        Ok(DocumentContent {
            raw_text,
            metadata,
            document_type,
            chunks,
        })
    }

    pub fn detect_document_type(&self, file_path: &Path) -> std::result::Result<DocumentType, ProcessingError> {
        // First try MIME type detection
        let mime_type = mime_guess::from_path(file_path).first_or_octet_stream();
        
        match mime_type.as_ref() {
            "application/pdf" => return Ok(DocumentType::Pdf),
            "application/epub+zip" => return Ok(DocumentType::Epub),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document" => {
                return Ok(DocumentType::Docx);
            },
            "text/markdown" => return Ok(DocumentType::Markdown),
            "text/plain" => return Ok(DocumentType::Text),
            _ => {}
        }
        
        // Fall back to extension-based detection
        if let Some(extension) = file_path.extension().and_then(|e| e.to_str()) {
            match extension.to_lowercase().as_str() {
                "pdf" => Ok(DocumentType::Pdf),
                "epub" => Ok(DocumentType::Epub),
                "docx" => Ok(DocumentType::Docx),
                "md" | "markdown" => Ok(DocumentType::Markdown),
                "txt" => Ok(DocumentType::Text),
                // Code file extensions
                "rs" => Ok(DocumentType::Code("rust".to_string())),
                "py" => Ok(DocumentType::Code("python".to_string())),
                "js" | "mjs" => Ok(DocumentType::Code("javascript".to_string())),
                "ts" => Ok(DocumentType::Code("typescript".to_string())),
                "json" => Ok(DocumentType::Code("json".to_string())),
                "yaml" | "yml" => Ok(DocumentType::Code("yaml".to_string())),
                "toml" => Ok(DocumentType::Code("toml".to_string())),
                "xml" | "html" | "htm" => Ok(DocumentType::Code("xml".to_string())),
                "c" | "h" => Ok(DocumentType::Code("c".to_string())),
                "cpp" | "cc" | "cxx" | "hpp" => Ok(DocumentType::Code("cpp".to_string())),
                "java" => Ok(DocumentType::Code("java".to_string())),
                "go" => Ok(DocumentType::Code("go".to_string())),
                "rb" => Ok(DocumentType::Code("ruby".to_string())),
                "php" => Ok(DocumentType::Code("php".to_string())),
                "sh" | "bash" => Ok(DocumentType::Code("bash".to_string())),
                "css" => Ok(DocumentType::Code("css".to_string())),
                "scss" | "sass" => Ok(DocumentType::Code("scss".to_string())),
                "sql" => Ok(DocumentType::Code("sql".to_string())),
                _ => Ok(DocumentType::Unknown),
            }
        } else {
            Ok(DocumentType::Unknown)
        }
    }

    async fn extract_pdf_text(&self, file_path: &Path) -> std::result::Result<String, ProcessingError> {
        // For now, return a placeholder implementation
        // TODO: Implement proper PDF parsing using pdf-extract or similar crate
        // The pdf = "0.8" crate has a complex API that needs careful integration
        
        tracing::warn!("PDF parsing not yet implemented, returning placeholder text");
        Ok(format!("PDF file: {} (content extraction not implemented)", file_path.display()))
    }

    async fn extract_epub_text(&self, file_path: &Path) -> std::result::Result<String, ProcessingError> {
        let mut doc = epub::doc::EpubDoc::new(file_path)
            .map_err(|e| ProcessingError::Parse(format!("Failed to parse EPUB: {}", e)))?;
        
        let mut text = String::new();
        let spine = doc.spine.clone();
        
        for spine_item in spine.iter() {
            if let Some((content, _media_type)) = doc.get_resource_str(&spine_item.idref) {
                // Simple HTML tag removal (for better text extraction, consider using html2text crate)
                let clean_content = self.strip_html_tags(&content);
                if !text.is_empty() && !text.ends_with('\n') {
                    text.push('\n');
                }
                text.push_str(&clean_content);
            }
        }
        
        Ok(text)
    }

    async fn extract_docx_text(&self, file_path: &Path) -> std::result::Result<String, ProcessingError> {
        use std::io::Read;
        
        let file = std::fs::File::open(file_path)
            .map_err(ProcessingError::Io)?;
        
        let mut archive = zip::ZipArchive::new(file)
            .map_err(|e| ProcessingError::Parse(format!("Failed to read DOCX archive: {}", e)))?;
        
        let mut document_xml = archive.by_name("word/document.xml")
            .map_err(|e| ProcessingError::Parse(format!("Failed to find document.xml: {}", e)))?;
        
        let mut xml_content = String::new();
        document_xml.read_to_string(&mut xml_content)
            .map_err(ProcessingError::Io)?;
        
        // Extract text from XML (basic implementation - removes XML tags)
        let text = self.extract_text_from_docx_xml(&xml_content);
        
        Ok(text)
    }

    async fn extract_text_file_content(&self, file_path: &Path) -> std::result::Result<String, ProcessingError> {
        let bytes = tokio::fs::read(file_path).await
            .map_err(ProcessingError::Io)?;
        
        // Detect encoding
        let encoding = chardet::detect(&bytes);
        let encoding_name = &encoding.0;
        
        // Convert to UTF-8
        let (decoded, _, had_errors) = encoding_rs::Encoding::for_label(encoding_name.as_bytes())
            .unwrap_or(encoding_rs::UTF_8)
            .decode(&bytes);
        
        if had_errors {
            tracing::warn!("Encoding conversion had errors for file: {}", file_path.display());
        }
        
        Ok(decoded.to_string())
    }

    async fn extract_code_file_content(&self, file_path: &Path) -> std::result::Result<String, ProcessingError> {
        // Extract raw text content
        let content = self.extract_text_file_content(file_path).await?;
        
        // Add tree-sitter parsing for better code structure understanding
        if let Some(extension) = file_path.extension().and_then(|e| e.to_str()) {
            match extension.to_lowercase().as_str() {
                "rs" => {
                    if let Ok(enhanced_content) = self.parse_with_tree_sitter(&content, "rust").await {
                        return Ok(enhanced_content);
                    }
                },
                "py" => {
                    if let Ok(enhanced_content) = self.parse_with_tree_sitter(&content, "python").await {
                        return Ok(enhanced_content);
                    }
                },
                "js" | "mjs" => {
                    if let Ok(enhanced_content) = self.parse_with_tree_sitter(&content, "javascript").await {
                        return Ok(enhanced_content);
                    }
                },
                "json" => {
                    if let Ok(enhanced_content) = self.parse_with_tree_sitter(&content, "json").await {
                        return Ok(enhanced_content);
                    }
                },
                _ => {}
            }
        }
        
        // Fall back to raw content if tree-sitter parsing fails
        Ok(content)
    }

    async fn parse_with_tree_sitter(&self, content: &str, language: &str) -> std::result::Result<String, ProcessingError> {
        use tree_sitter::Parser;
        
        // Get the appropriate language parser
        let language_fn = match language {
            "rust" => tree_sitter_rust::language,
            "python" => tree_sitter_python::language,
            "javascript" => tree_sitter_javascript::language,
            "json" => tree_sitter_json::language,
            _ => return Err(ProcessingError::Parse(format!("Unsupported language: {}", language))),
        };
        
        let mut parser = Parser::new();
        parser.set_language(language_fn())
            .map_err(|e| ProcessingError::Parse(format!("Failed to set parser language: {}", e)))?;
        
        let tree = parser.parse(content, None)
            .ok_or_else(|| ProcessingError::Parse("Failed to parse code".to_string()))?;
        
        // Extract structured information from the parse tree
        let mut enhanced_content = String::new();
        enhanced_content.push_str("=== CODE STRUCTURE ===\n");
        
        // Add the original content
        enhanced_content.push_str(content);
        enhanced_content.push_str("\n\n=== PARSED STRUCTURE ===\n");
        
        // Walk the tree and extract meaningful nodes
        let root_node = tree.root_node();
        Self::extract_code_structure(&mut enhanced_content, root_node, content, 0);
        
        Ok(enhanced_content)
    }

    fn extract_code_structure(output: &mut String, node: tree_sitter::Node, source: &str, depth: usize) {
        let indent = "  ".repeat(depth);
        
        // Extract text for the node
        let node_text = node.utf8_text(source.as_bytes()).unwrap_or("[invalid utf8]");
        let node_kind = node.kind();
        
        // Only include meaningful structural elements
        match node_kind {
            "function_item" | "function_declaration" | "function_definition" |
            "method_definition" | "class_definition" | "struct_item" | "enum_item" |
            "impl_item" | "trait_item" | "module_item" | "use_declaration" |
            "import_statement" | "export_statement" => {
                
                // Extract the first line or a summary
                let first_line = node_text.lines().next().unwrap_or(node_text);
                let summary = if first_line.len() > 100 {
                    format!("{}...", &first_line[..97])
                } else {
                    first_line.to_string()
                };
                
                output.push_str(&format!("{}[{}] {}\n", indent, node_kind, summary));
            },
            "comment" | "line_comment" | "block_comment" => {
                // Include important comments
                if node_text.contains("TODO") || node_text.contains("FIXME") || 
                   node_text.contains("NOTE") || node_text.contains("WARNING") {
                    output.push_str(&format!("{}[{}] {}\n", indent, node_kind, node_text.trim()));
                }
            },
            _ => {
                // For other node types, recurse without printing
            }
        }
        
        // Recurse into children for structural elements
        if matches!(node_kind, 
            "source_file" | "program" | "module" | "class_body" | "block" | "declaration_list"
        ) || depth < 3 {
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i) {
                    Self::extract_code_structure(output, child, source, depth + 1);
                }
            }
        }
    }

    fn create_text_chunks(&self, text: &str, document_type: &DocumentType) -> std::result::Result<Vec<TextChunk>, ProcessingError> {
        let mut chunks = Vec::new();
        
        if text.is_empty() {
            return Ok(chunks);
        }
        
        // Simple token-based chunking (approximate)
        // In a production system, you'd want proper tokenization
        let words: Vec<&str> = text.split_whitespace().collect();
        
        if words.is_empty() {
            return Ok(chunks);
        }
        
        let mut current_chunk_start = 0;
        let mut chunk_index = 0;
        
        while current_chunk_start < words.len() {
            let chunk_end = std::cmp::min(
                current_chunk_start + self.chunking_config.chunk_size,
                words.len()
            );
            
            let chunk_words = &words[current_chunk_start..chunk_end];
            let chunk_text = chunk_words.join(" ");
            
            // Calculate character positions (approximate)
            let start_char = if current_chunk_start == 0 {
                0
            } else {
                words[0..current_chunk_start].join(" ").len() + 1
            };
            
            let end_char = start_char + chunk_text.len();
            
            let mut metadata = HashMap::new();
            metadata.insert("chunk_type".to_string(), "text".to_string());
            metadata.insert("document_type".to_string(), format!("{:?}", document_type));
            metadata.insert("word_count".to_string(), chunk_words.len().to_string());
            
            chunks.push(TextChunk {
                content: chunk_text,
                chunk_index,
                start_char,
                end_char,
                metadata,
            });
            
            chunk_index += 1;
            
            // Calculate next chunk start with overlap
            if chunk_end >= words.len() {
                break;
            }
            
            current_chunk_start = if chunk_end > self.chunking_config.overlap_size {
                chunk_end - self.chunking_config.overlap_size
            } else {
                chunk_end
            };
        }
        
        Ok(chunks)
    }

    fn strip_html_tags(&self, html: &str) -> String {
        // Basic HTML tag removal - in production, consider using html2text crate
        let mut result = String::new();
        let mut in_tag = false;
        
        for ch in html.chars() {
            match ch {
                '<' => in_tag = true,
                '>' => in_tag = false,
                _ if !in_tag => result.push(ch),
                _ => {}
            }
        }
        
        // Clean up excessive whitespace
        result.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    fn extract_text_from_docx_xml(&self, xml: &str) -> String {
        // Basic XML text extraction for DOCX document.xml
        // Look for text within <w:t> tags
        let mut result = String::new();
        let mut in_text_tag = false;
        let mut current_tag = String::new();
        
        let mut chars = xml.chars().peekable();
        
        while let Some(ch) = chars.next() {
            match ch {
                '<' => {
                    current_tag.clear();
                    current_tag.push(ch);
                    
                    // Read the full tag
                    while let Some(&next_ch) = chars.peek() {
                        current_tag.push(chars.next().unwrap());
                        if next_ch == '>' {
                            break;
                        }
                    }
                    
                    // Check if this is a text tag
                    if current_tag.starts_with("<w:t") {
                        in_text_tag = true;
                    } else if current_tag.starts_with("</w:t>") {
                        in_text_tag = false;
                        result.push(' '); // Add space between text runs
                    }
                },
                _ if in_text_tag => {
                    result.push(ch);
                },
                _ => {}
            }
        }
        
        // Clean up excessive whitespace
        result.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    /// Check if the document processor is healthy
    pub async fn is_healthy(&self) -> bool {
        // Check if we can create a simple text chunk
        match self.create_text_chunks("test", &DocumentType::Text) {
            Ok(_) => true,
            Err(e) => {
                tracing::warn!("DocumentProcessor health check failed: {}", e);
                false
            }
        }
    }

    /// Test Qdrant connection (placeholder implementation)
    pub async fn test_qdrant_connection(&self) -> std::result::Result<(), ProcessingError> {
        // TODO: Implement actual Qdrant connection test
        // For now, just return success as a placeholder
        tracing::debug!("Qdrant connection test - placeholder implementation");
        Ok(())
    }

    /// Get processing statistics
    pub fn get_stats(&self) -> ProcessingStats {
        ProcessingStats {
            total_documents_processed: 0, // TODO: Implement actual tracking
            total_chunks_created: 0,
            average_processing_time_ms: 0,
            chunking_config: self.chunking_config.clone(),
        }
    }
}

impl Default for DocumentProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Unified processing engine that integrates all components
pub struct ProcessingEngine {
    /// Priority-based task processing pipeline
    pipeline: Arc<Mutex<Pipeline>>,
    /// Task submitter for external requests
    task_submitter: TaskSubmitter,
    /// IPC server for Python communication
    ipc_server: Option<IpcServer>,
    /// Storage client for Qdrant operations
    #[allow(dead_code)]
    storage_client: Arc<StorageClient>,
    /// Document processor
    #[allow(dead_code)]
    document_processor: Arc<DocumentProcessor>,
    /// Daemon state manager for SQLite persistence
    daemon_state_manager: Option<Arc<DaemonStateManager>>,
    /// Engine configuration
    config: Arc<Config>,
}

impl ProcessingEngine {
    /// Create a new processing engine with default configuration
    pub fn new() -> Self {
        let pipeline = Arc::new(Mutex::new(Pipeline::new(4))); // Default 4 concurrent tasks
        let task_submitter = {
            let pipeline_lock = pipeline.try_lock().unwrap();
            pipeline_lock.task_submitter()
        };
        
        Self {
            pipeline,
            task_submitter,
            ipc_server: None,
            storage_client: Arc::new(StorageClient::new()),
            document_processor: Arc::new(DocumentProcessor::new()),
            daemon_state_manager: None,
            config: Arc::new(Config::default()),
        }
    }
    
    /// Create a processing engine with custom daemon configuration
    pub fn with_daemon_config(daemon_config: DaemonConfig) -> Self {
        let config = Config::from(daemon_config.clone());
        let max_concurrent = config.max_concurrent_tasks.unwrap_or(4);
        let pipeline = Arc::new(Mutex::new(Pipeline::new(max_concurrent)));
        let task_submitter = {
            let pipeline_lock = pipeline.try_lock().unwrap();
            pipeline_lock.task_submitter()
        };
        
        // Create storage client with the Qdrant configuration from daemon config
        let storage_client = Arc::new(StorageClient::with_config(daemon_config.qdrant));
        
        Self {
            pipeline,
            task_submitter,
            ipc_server: None,
            storage_client,
            document_processor: Arc::new(DocumentProcessor::new()),
            daemon_state_manager: None,
            config: Arc::new(config),
        }
    }
    
    /// Create a processing engine with unified configuration (supports TOML/YAML)
    pub fn with_unified_config(config_file: Option<&Path>, config_dir: Option<&Path>) -> std::result::Result<Self, UnifiedConfigError> {
        let config_manager = UnifiedConfigManager::new(config_dir);
        let daemon_config = config_manager.load_config(config_file)?;
        
        tracing::info!("Loaded configuration using unified config manager");
        Ok(Self::with_daemon_config(daemon_config))
    }
    
    /// Create a processing engine with unified configuration and auto-discovery
    pub fn from_unified_config() -> std::result::Result<Self, UnifiedConfigError> {
        Self::with_unified_config(None, None)
    }
    
    /// Create a processing engine with custom configuration
    pub fn with_config(config: Config) -> Self {
        let max_concurrent = config.max_concurrent_tasks.unwrap_or(4);
        let pipeline = Arc::new(Mutex::new(Pipeline::new(max_concurrent)));
        let task_submitter = {
            let pipeline_lock = pipeline.try_lock().unwrap();
            pipeline_lock.task_submitter()
        };
        
        Self {
            pipeline,
            task_submitter,
            ipc_server: None,
            storage_client: Arc::new(StorageClient::new()),
            document_processor: Arc::new(DocumentProcessor::new()),
            daemon_state_manager: None,
            config: Arc::new(config),
        }
    }
    
    /// Initialize daemon state manager with proper database location
    async fn initialize_daemon_state_manager(&mut self) -> std::result::Result<(), ProcessingError> {
        use std::env;
        
        // Determine database location based on platform and user environment
        let db_path = if let Ok(home) = env::var("HOME") {
            // Use user's home directory for database storage
            std::path::PathBuf::from(home)
                .join(".local")
                .join("share")
                .join("workspace-qdrant")
                .join("state.db")
        } else {
            // Fallback to temporary directory
            std::env::temp_dir()
                .join("workspace-qdrant")
                .join("state.db")
        };
        
        tracing::info!("Initializing daemon state manager with database: {}", db_path.display());
        
        // Create the daemon state manager
        let state_manager = DaemonStateManager::new(&db_path).await
            .map_err(|e| ProcessingError::Processing(format!("Failed to create daemon state manager: {}", e)))?;
        
        // Initialize the database schema
        state_manager.initialize().await
            .map_err(|e| ProcessingError::Processing(format!("Failed to initialize daemon database schema: {}", e)))?;
        
        self.daemon_state_manager = Some(Arc::new(state_manager));
        
        tracing::info!("Daemon state manager initialized successfully");
        Ok(())
    }
    
    /// Start the processing engine with IPC support
    pub async fn start_with_ipc(&mut self) -> std::result::Result<IpcClient, ProcessingError> {
        // Initialize daemon state manager first
        self.initialize_daemon_state_manager().await?;
        
        // Start the main pipeline
        {
            let mut pipeline_lock = self.pipeline.lock().await;
            pipeline_lock.start().await
                .map_err(|e| ProcessingError::Processing(e.to_string()))?;
        }
        
        // Create and start IPC server
        let max_concurrent = self.config.max_concurrent_tasks.unwrap_or(4);
        let (ipc_server, ipc_client) = IpcServer::new(max_concurrent);
        
        ipc_server.start().await
            .map_err(|e| ProcessingError::Processing(e.to_string()))?;
        
        self.ipc_server = Some(ipc_server);
        
        tracing::info!("Processing engine started with IPC support");
        Ok(ipc_client)
    }
    
    /// Start the processing engine without IPC (standalone mode)
    pub async fn start(&mut self) -> std::result::Result<(), ProcessingError> {
        // Initialize daemon state manager first
        self.initialize_daemon_state_manager().await?;
        
        let mut pipeline_lock = self.pipeline.lock().await;
        pipeline_lock.start().await
            .map_err(|e| ProcessingError::Processing(e.to_string()))?;
        
        tracing::info!("Processing engine started in standalone mode");
        Ok(())
    }
    
    /// Submit a document processing task
    pub async fn process_document(
        &self,
        file_path: &Path,
        collection: &str,
        priority: TaskPriority,
    ) -> std::result::Result<TaskResult, ProcessingError> {
        let source = match priority {
            TaskPriority::McpRequests => TaskSource::McpServer {
                request_id: uuid::Uuid::new_v4().to_string(),
            },
            TaskPriority::ProjectWatching => TaskSource::ProjectWatcher {
                project_path: file_path.parent()
                    .unwrap_or_else(|| Path::new("/"))
                    .to_string_lossy()
                    .to_string(),
            },
            TaskPriority::CliCommands => TaskSource::CliCommand {
                command: format!("process-document {}", file_path.display()),
            },
            TaskPriority::BackgroundWatching => TaskSource::BackgroundWatcher {
                folder_path: file_path.parent()
                    .unwrap_or_else(|| Path::new("/"))
                    .to_string_lossy()
                    .to_string(),
            },
        };
        
        let payload = TaskPayload::ProcessDocument {
            file_path: file_path.to_path_buf(),
            collection: collection.to_string(),
        };
        
        let timeout = self.config.default_timeout_ms
            .map(Duration::from_millis);
        
        let task_handle = self.task_submitter
            .submit_task(priority, source, payload, timeout)
            .await
            .map_err(|e| ProcessingError::Processing(e.to_string()))?;
        
        task_handle.wait().await
            .map_err(|e| ProcessingError::Processing(e.to_string()))
    }
    
    /// Submit a directory watching task
    pub async fn watch_directory(
        &self,
        path: &Path,
        recursive: bool,
        priority: TaskPriority,
    ) -> std::result::Result<TaskResult, ProcessingError> {
        let source = match priority {
            TaskPriority::ProjectWatching => TaskSource::ProjectWatcher {
                project_path: path.to_string_lossy().to_string(),
            },
            TaskPriority::BackgroundWatching => TaskSource::BackgroundWatcher {
                folder_path: path.to_string_lossy().to_string(),
            },
            _ => TaskSource::CliCommand {
                command: format!("watch-directory {}", path.display()),
            },
        };
        
        let payload = TaskPayload::WatchDirectory {
            path: path.to_path_buf(),
            recursive,
        };
        
        let timeout = self.config.default_timeout_ms
            .map(Duration::from_millis);
        
        let task_handle = self.task_submitter
            .submit_task(priority, source, payload, timeout)
            .await
            .map_err(|e| ProcessingError::Processing(e.to_string()))?;
        
        task_handle.wait().await
            .map_err(|e| ProcessingError::Processing(e.to_string()))
    }
    
    /// Execute a search query
    pub async fn execute_query(
        &self,
        query: &str,
        collection: &str,
        limit: usize,
        priority: TaskPriority,
    ) -> std::result::Result<TaskResult, ProcessingError> {
        let source = TaskSource::McpServer {
            request_id: uuid::Uuid::new_v4().to_string(),
        };
        
        let payload = TaskPayload::ExecuteQuery {
            query: query.to_string(),
            collection: collection.to_string(),
            limit,
        };
        
        let timeout = Some(Duration::from_millis(5000)); // Queries should be fast
        
        let task_handle = self.task_submitter
            .submit_task(priority, source, payload, timeout)
            .await
            .map_err(|e| ProcessingError::Processing(e.to_string()))?;
        
        task_handle.wait().await
            .map_err(|e| ProcessingError::Processing(e.to_string()))
    }
    
    /// Get pipeline statistics
    pub async fn get_stats(&self) -> std::result::Result<processing::PipelineStats, ProcessingError> {
        let pipeline_lock = self.pipeline.lock().await;
        Ok(pipeline_lock.stats().await)
    }
    
    /// Get task submitter for advanced usage
    pub fn task_submitter(&self) -> TaskSubmitter {
        self.task_submitter.clone()
    }
    
    /// Get daemon state manager if initialized
    pub fn daemon_state_manager(&self) -> Option<Arc<DaemonStateManager>> {
        self.daemon_state_manager.clone()
    }
    
    /// Graceful shutdown
    pub async fn shutdown(&mut self) -> std::result::Result<(), ProcessingError> {
        if let Some(ipc_server) = &self.ipc_server {
            // Wait for IPC server to shutdown
            ipc_server.wait_for_shutdown().await;
        }
        
        tracing::info!("Processing engine shutdown complete");
        Ok(())
    }
}

impl Default for ProcessingEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Basic health check function
pub fn health_check() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_check() {
        assert!(health_check());
    }

    #[tokio::test]
    async fn test_document_processor() {
        let processor = DocumentProcessor::new();
        // Basic instantiation test
        assert!(processor
            .process_file(Path::new("/tmp/test.txt"), "test")
            .await
            .is_err());
    }
}
