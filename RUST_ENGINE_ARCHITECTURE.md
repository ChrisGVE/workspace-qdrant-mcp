# Rust Ingestion Engine Architecture v2.0

**Version:** 2.0  
**Date:** 2025-08-30  
**Status:** Complete Architecture Specification

## Executive Summary

This document defines the complete architecture for the Rust-based ingestion engine that forms the processing core of workspace-qdrant-mcp v2.0. The engine implements a two-process architecture with the Python MCP server handling interface operations and the Rust engine performing heavy processing tasks.

The Rust engine is embedded within the Python package using maturin/setuptools-rust, enabling PyPI distribution with pre-built wheels for Tier 1 platforms while maintaining high performance through native compilation.

## Strategic Architecture Vision

### Core Design Principles

1. **Embedded Deployment**: Rust engine built as a Python extension and distributed via PyPI
2. **gRPC Communication**: High-performance binary protocol between Python MCP and Rust engine
3. **Graceful Lifecycle**: Engine starts on MCP load, finishes work before shutdown
4. **Zero-Copy Processing**: Minimize memory allocation during file processing
5. **Platform Compatibility**: Pre-built wheels for major platforms with source fallback
6. **Extensible Processing**: Plugin-like architecture for different file formats

## Architecture Overview

### Two-Process Design

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface Layer                     │
├─────────────────────────────────────────────────────────────────┤
│  Claude Code ←→ Python MCP Server (FastMCP)                    │
│  • MCP Tools (11 existing tools)                               │
│  • Search Interface & Research Modes                           │
│  • Memory System Integration                                   │
│  • Collection Management                                       │
│  • Real-time Query Response (<100ms)                          │
└─────────────────────────────────────────────────────────────────┘
                              ↕ gRPC Protocol
┌─────────────────────────────────────────────────────────────────┐
│                      Processing Engine Layer                   │
├─────────────────────────────────────────────────────────────────┤
│  Rust Ingestion Engine (Embedded Binary)                      │
│  • Document Processing Pipeline                                │
│  • File Watching & Auto-ingestion                            │
│  • Embedding Generation                                       │
│  • LSP Integration                                            │
│  • Knowledge Graph Construction                               │
│  • Background Task Queue                                      │
└─────────────────────────────────────────────────────────────────┘
                              ↕ Qdrant Client
┌─────────────────────────────────────────────────────────────────┐
│                       Vector Storage Layer                     │
├─────────────────────────────────────────────────────────────────┤
│  Qdrant Vector Database                                        │
│  • Collection Storage (memory, _library, project-*)           │
│  • Vector & Metadata Indexing                                 │
│  • Hybrid Search (Dense + Sparse)                             │
└─────────────────────────────────────────────────────────────────┘
```

### Communication Protocol

The Python MCP and Rust engine communicate via gRPC using a custom protocol optimized for document processing workloads:

```protobuf
// Core service definition
service IngestionEngine {
    // Engine lifecycle
    rpc StartEngine(StartEngineRequest) returns (EngineStatus);
    rpc GetEngineStatus(Empty) returns (EngineStatus);
    rpc StopEngine(StopEngineRequest) returns (EngineStatus);
    
    // Document processing
    rpc IngestDocument(IngestDocumentRequest) returns (stream IngestProgress);
    rpc IngestFolder(IngestFolderRequest) returns (stream IngestProgress);
    rpc ProcessYamlMetadata(ProcessYamlRequest) returns (stream IngestProgress);
    
    // File watching
    rpc StartWatching(WatchRequest) returns (WatchStatus);
    rpc StopWatching(StopWatchRequest) returns (WatchStatus);
    rpc GetWatchStatus(Empty) returns (WatchStatus);
}
```

## Rust Engine Core Architecture

### Module Structure

```
rust-engine/
├── Cargo.toml                    # Dependencies and build config
├── src/
│   ├── main.rs                   # Binary entry point
│   ├── lib.rs                    # Python binding entry point
│   ├── grpc/
│   │   ├── mod.rs               # gRPC server implementation
│   │   ├── ingestion.proto      # Protocol definitions
│   │   └── services.rs          # Service implementations
│   ├── processing/
│   │   ├── mod.rs               # Document processing coordinator
│   │   ├── pipeline.rs          # Processing pipeline
│   │   ├── formats/             # File format processors
│   │   │   ├── mod.rs
│   │   │   ├── text.rs          # Plain text processor
│   │   │   ├── pdf.rs           # PDF processor
│   │   │   ├── epub.rs          # EPUB/MOBI processor
│   │   │   ├── code.rs          # Code file processor
│   │   │   └── web.rs           # Web page processor
│   │   └── metadata.rs          # Metadata extraction
│   ├── watching/
│   │   ├── mod.rs               # File watching coordinator
│   │   ├── watcher.rs           # Cross-platform file watcher
│   │   └── debouncer.rs         # Event debouncing
│   ├── embeddings/
│   │   ├── mod.rs               # Embedding generation
│   │   ├── local_models.rs      # Local embedding models
│   │   └── chunking.rs          # Document chunking strategies
│   ├── lsp/
│   │   ├── mod.rs               # LSP integration
│   │   ├── client.rs            # LSP client implementation  
│   │   ├── detection.rs         # Language server auto-detection
│   │   └── analysis.rs          # Code analysis enhancement
│   ├── storage/
│   │   ├── mod.rs               # Storage abstraction
│   │   └── qdrant.rs            # Qdrant client operations
│   └── config/
│       ├── mod.rs               # Configuration management
│       └── validation.rs        # Config validation
├── proto/
│   └── ingestion.proto          # gRPC protocol definitions
├── python-bindings/
│   ├── lib.rs                   # PyO3 Python bindings
│   └── engine.py                # Python wrapper classes
└── build.rs                     # Build script for proto generation
```

### Core Dependencies

```toml
[dependencies]
# Async runtime
tokio = { version = "1.35", features = ["full"] }
tokio-util = "0.7"

# gRPC communication
tonic = "0.10"
tonic-build = "0.10"
prost = "0.12"

# ML and embeddings
candle-core = "0.6"
candle-nn = "0.6"
candle-transformers = "0.6"
# Alternative: ort = "1.16" (ONNX Runtime)

# Document processing
pdf = "0.8"
epub = "2.0"
tree-sitter = "0.20"
tree-sitter-rust = "0.20"
tree-sitter-python = "0.20"
tree-sitter-javascript = "0.20"
tree-sitter-typescript = "0.20"

# File watching
notify = "6.1"
debounce = "0.2"

# Storage and serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
serde_yaml = "0.9"
qdrant-client = "1.7"

# LSP integration
tower-lsp = "0.20"
lsp-types = "0.94"

# Configuration and CLI
clap = { version = "4.0", features = ["derive"] }
config = "0.14"

# Utilities
uuid = { version = "1.0", features = ["v4"] }
chrono = { version = "0.4", features = ["serde"] }
thiserror = "1.0"
anyhow = "1.0"
tracing = "0.1"
tracing-subscriber = "0.3"

# Python bindings (for embedded mode)
pyo3 = { version = "0.20", features = ["extension-module"], optional = true }

[build-dependencies]
tonic-build = "0.10"

[features]
default = ["python-bindings"]
python-bindings = ["pyo3"]
standalone = []
```

## Embedded Deployment Strategy

### Maturin Integration

The Rust engine is compiled as a Python extension module using maturin, enabling seamless embedding within the Python package:

```toml
# pyproject.toml additions
[build-system]
requires = ["maturin>=1.4,<2.0"]
build-backend = "maturin"

[tool.maturin]
python-source = "python"
module-name = "workspace_qdrant_mcp._rust_engine"
features = ["python-bindings"]

# Platform-specific builds
[tool.maturin.target.x86_64-apple-darwin]
# macOS Intel optimizations

[tool.maturin.target.aarch64-apple-darwin]
# macOS Apple Silicon optimizations

[tool.maturin.target.x86_64-pc-windows-msvc]
# Windows x64 optimizations

[tool.maturin.target.aarch64-pc-windows-msvc]
# Windows ARM64 optimizations

[tool.maturin.target.x86_64-unknown-linux-gnu]
# Linux x86_64 optimizations
```

### Python Integration Layer

```python
# src/workspace_qdrant_mcp/rust_engine/__init__.py
"""
Rust ingestion engine Python interface.

This module provides the Python wrapper for the embedded Rust ingestion engine,
handling process lifecycle, gRPC communication, and error translation.
"""

import asyncio
import atexit
import logging
from pathlib import Path
from typing import Dict, List, Optional, AsyncIterator

import grpc
from grpc import aio as aio_grpc

from ._rust_engine import RustIngestionEngine  # Compiled Rust extension
from .proto import ingestion_pb2, ingestion_pb2_grpc
from .config import EngineConfig

logger = logging.getLogger(__name__)

class IngestionEngineManager:
    """Manages the lifecycle of the embedded Rust ingestion engine."""
    
    def __init__(self, config: EngineConfig):
        self.config = config
        self.engine: Optional[RustIngestionEngine] = None
        self.grpc_channel: Optional[aio_grpc.Channel] = None
        self.grpc_client: Optional[ingestion_pb2_grpc.IngestionEngineStub] = None
        self._shutdown_registered = False
        
    async def start(self) -> bool:
        """Start the Rust engine and establish gRPC communication."""
        try:
            # Initialize embedded Rust engine
            self.engine = RustIngestionEngine(self.config.to_dict())
            await self.engine.start()
            
            # Establish gRPC channel
            self.grpc_channel = aio_grpc.insecure_channel(
                f"127.0.0.1:{self.engine.grpc_port()}"
            )
            self.grpc_client = ingestion_pb2_grpc.IngestionEngineStub(
                self.grpc_channel
            )
            
            # Register shutdown handler
            if not self._shutdown_registered:
                atexit.register(self._sync_shutdown)
                self._shutdown_registered = True
            
            logger.info(f"Rust ingestion engine started on port {self.engine.grpc_port()}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Rust engine: {e}")
            await self._cleanup()
            return False
    
    async def ingest_document(
        self, 
        file_path: str, 
        collection: str, 
        metadata: Optional[Dict] = None
    ) -> AsyncIterator[Dict]:
        """Ingest a single document through the Rust engine."""
        if not self.grpc_client:
            raise RuntimeError("Engine not started")
        
        request = ingestion_pb2.IngestDocumentRequest(
            file_path=file_path,
            collection=collection,
            metadata=metadata or {}
        )
        
        async for progress in self.grpc_client.IngestDocument(request):
            yield {
                "status": progress.status,
                "progress": progress.progress,
                "message": progress.message,
                "error": progress.error if progress.error else None
            }
```

### Platform Distribution

**Tier 1 Platforms** (Pre-built wheels via CI/CD):
- macOS Intel (x86_64-apple-darwin)
- macOS Apple Silicon (aarch64-apple-darwin) 
- Linux x86_64 (x86_64-unknown-linux-gnu)
- Windows x64 (x86_64-pc-windows-msvc)
- Windows ARM64 (aarch64-pc-windows-msvc)

**CI/CD Pipeline** (GitHub Actions):
```yaml
# .github/workflows/rust-wheels.yml
name: Build Rust Wheels

on:
  push:
    tags: ['v*']
  workflow_dispatch:

jobs:
  build-wheels:
    strategy:
      matrix:
        include:
          - os: macos-latest
            target: x86_64-apple-darwin
          - os: macos-latest 
            target: aarch64-apple-darwin
          - os: ubuntu-latest
            target: x86_64-unknown-linux-gnu
          - os: windows-latest
            target: x86_64-pc-windows-msvc
          - os: windows-latest
            target: aarch64-pc-windows-msvc
    
    runs-on: ${{ matrix.os }}
    
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable
        with:
          targets: ${{ matrix.target }}
      
      - name: Install maturin
        run: pip install maturin
      
      - name: Build wheel
        run: maturin build --release --target ${{ matrix.target }}
      
      - name: Upload wheel
        uses: actions/upload-artifact@v3
        with:
          name: wheels-${{ matrix.target }}
          path: target/wheels/*.whl
```

## Document Processing Pipeline

### Processing Architecture

```rust
// src/processing/pipeline.rs
use std::path::Path;
use tokio::sync::mpsc;
use tracing::{info, error};

pub struct ProcessingPipeline {
    // Format processors
    text_processor: TextProcessor,
    pdf_processor: PdfProcessor,
    epub_processor: EpubProcessor,
    code_processor: CodeProcessor,
    web_processor: WebProcessor,
    
    // Embedding generation
    embedder: EmbeddingGenerator,
    
    // Storage interface
    storage: StorageClient,
}

impl ProcessingPipeline {
    pub async fn process_document(
        &self,
        file_path: &Path,
        collection: &str,
        metadata: DocumentMetadata,
        progress_tx: mpsc::Sender<ProcessingProgress>,
    ) -> Result<DocumentResult, ProcessingError> {
        
        // Stage 1: Format detection and basic metadata
        let file_info = self.analyze_file(file_path).await?;
        progress_tx.send(ProcessingProgress::stage("File analyzed")).await?;
        
        // Stage 2: Content extraction
        let document = match file_info.format {
            FileFormat::Text => self.text_processor.process(file_path).await?,
            FileFormat::Pdf => self.pdf_processor.process(file_path).await?,
            FileFormat::Epub => self.epub_processor.process(file_path).await?,
            FileFormat::Code(lang) => self.code_processor.process(file_path, lang).await?,
            FileFormat::Web => self.web_processor.process(file_path).await?,
        };
        progress_tx.send(ProcessingProgress::stage("Content extracted")).await?;
        
        // Stage 3: LSP enhancement (for code files)
        let enhanced_document = if matches!(file_info.format, FileFormat::Code(_)) {
            self.enhance_with_lsp(document).await?
        } else {
            document
        };
        progress_tx.send(ProcessingProgress::stage("LSP analysis complete")).await?;
        
        // Stage 4: Chunking and embedding generation
        let chunks = self.chunk_document(enhanced_document).await?;
        let embeddings = self.embedder.generate_embeddings(&chunks).await?;
        progress_tx.send(ProcessingProgress::stage("Embeddings generated")).await?;
        
        // Stage 5: Storage in Qdrant
        let result = self.storage.store_document(
            collection,
            chunks,
            embeddings,
            metadata,
        ).await?;
        progress_tx.send(ProcessingProgress::completed(result.clone())).await?;
        
        Ok(result)
    }
    
    async fn chunk_document(
        &self, 
        document: ProcessedDocument
    ) -> Result<Vec<DocumentChunk>, ProcessingError> {
        // Implement intelligent chunking based on document type
        // - Code: function/class boundaries
        // - Text: paragraph/section boundaries  
        // - PDF: page/section boundaries
        // - EPUB: chapter boundaries
        
        match document.document_type {
            DocumentType::Code => self.chunk_code_document(document).await,
            DocumentType::Text => self.chunk_text_document(document).await,
            DocumentType::Pdf => self.chunk_pdf_document(document).await,
            DocumentType::Epub => self.chunk_epub_document(document).await,
        }
    }
}
```

### File Format Processors

#### PDF Processor
```rust
// src/processing/formats/pdf.rs
use pdf::file::File as PdfFile;
use pdf::primitive::Primitive;

pub struct PdfProcessor {
    // OCR integration for image-heavy PDFs
    ocr_enabled: bool,
}

impl PdfProcessor {
    pub async fn process(&self, file_path: &Path) -> Result<ProcessedDocument, ProcessingError> {
        let file = PdfFile::open(file_path)?;
        let mut content = String::new();
        let mut metadata = DocumentMetadata::default();
        
        // Extract text content
        for page_num in 1..=file.num_pages() {
            let page = file.get_page(page_num)?;
            let text = self.extract_page_text(&page).await?;
            
            if text.trim().is_empty() && self.ocr_enabled {
                // Fallback to OCR for image-based PDFs
                let ocr_text = self.ocr_page(&page).await?;
                content.push_str(&ocr_text);
            } else {
                content.push_str(&text);
            }
            content.push('\n');
        }
        
        // Extract metadata
        if let Some(info) = file.trailer.info_dict {
            metadata.title = info.get("Title").and_then(|p| p.as_string());
            metadata.author = info.get("Author").and_then(|p| p.as_string());
            metadata.subject = info.get("Subject").and_then(|p| p.as_string());
        }
        
        Ok(ProcessedDocument {
            content,
            metadata,
            document_type: DocumentType::Pdf,
            original_path: file_path.to_path_buf(),
        })
    }
}
```

#### Code Processor with LSP Integration
```rust
// src/processing/formats/code.rs
use tree_sitter::{Language, Parser, Tree};
use crate::lsp::{LspClient, SymbolInformation};

pub struct CodeProcessor {
    parsers: HashMap<String, Parser>,
    lsp_client: Option<LspClient>,
}

impl CodeProcessor {
    pub async fn process(
        &self, 
        file_path: &Path,
        language: CodeLanguage
    ) -> Result<ProcessedDocument, ProcessingError> {
        let content = tokio::fs::read_to_string(file_path).await?;
        
        // Parse with Tree-sitter
        let mut parser = self.get_parser(language)?;
        let tree = parser.parse(&content, None)
            .ok_or(ProcessingError::ParseError)?;
        
        // Extract structure and symbols
        let structure = self.extract_code_structure(&tree, &content).await?;
        
        // Enhance with LSP if available
        let enhanced_structure = if let Some(lsp) = &self.lsp_client {
            lsp.enhance_symbols(file_path, structure).await?
        } else {
            structure
        };
        
        Ok(ProcessedDocument {
            content,
            metadata: DocumentMetadata {
                language: Some(language.to_string()),
                symbols: enhanced_structure.symbols,
                imports: enhanced_structure.imports,
                ..Default::default()
            },
            document_type: DocumentType::Code,
            original_path: file_path.to_path_buf(),
        })
    }
    
    async fn extract_code_structure(
        &self,
        tree: &Tree,
        source: &str
    ) -> Result<CodeStructure, ProcessingError> {
        let mut structure = CodeStructure::default();
        let root_node = tree.root_node();
        
        // Walk the syntax tree to extract:
        // - Function definitions
        // - Class definitions  
        // - Import statements
        // - Comments and docstrings
        // - Variable declarations
        
        self.walk_node(root_node, source, &mut structure).await?;
        Ok(structure)
    }
}
```

## File Watching System

### Cross-Platform File Watcher

```rust
// src/watching/watcher.rs
use notify::{Config, Event, RecommendedWatcher, RecursiveMode, Watcher};
use std::collections::HashMap;
use tokio::sync::mpsc;

pub struct FileWatchingSystem {
    watcher: RecommendedWatcher,
    watched_paths: HashMap<PathBuf, WatchConfig>,
    event_sender: mpsc::Sender<FileEvent>,
    debouncer: EventDebouncer,
}

pub struct WatchConfig {
    pub collection: String,
    pub auto_ingest: bool,
    pub file_patterns: Vec<String>,
    pub ignore_patterns: Vec<String>,
}

impl FileWatchingSystem {
    pub async fn start_watching(
        &mut self,
        path: PathBuf,
        config: WatchConfig,
    ) -> Result<(), WatchError> {
        
        // Add to notify watcher
        self.watcher.watch(&path, RecursiveMode::Recursive)?;
        
        // Store configuration
        self.watched_paths.insert(path.clone(), config);
        
        info!("Started watching path: {}", path.display());
        Ok(())
    }
    
    pub async fn stop_watching(&mut self, path: &Path) -> Result<(), WatchError> {
        self.watcher.unwatch(path)?;
        self.watched_paths.remove(path);
        info!("Stopped watching path: {}", path.display());
        Ok(())
    }
    
    async fn handle_file_event(&self, event: Event) -> Result<(), WatchError> {
        // Debounce rapid file changes
        let debounced_event = self.debouncer.process_event(event).await?;
        
        if let Some(stable_event) = debounced_event {
            match stable_event.kind {
                EventKind::Create(_) | EventKind::Modify(_) => {
                    for path in stable_event.paths {
                        if let Some(config) = self.find_watch_config(&path) {
                            if self.should_process_file(&path, config) {
                                let file_event = FileEvent {
                                    path: path.clone(),
                                    event_type: FileEventType::Changed,
                                    collection: config.collection.clone(),
                                };
                                
                                self.event_sender.send(file_event).await?;
                            }
                        }
                    }
                }
                EventKind::Remove(_) => {
                    // Handle file removal
                    for path in stable_event.paths {
                        if let Some(config) = self.find_watch_config(&path) {
                            let file_event = FileEvent {
                                path: path.clone(),
                                event_type: FileEventType::Deleted,
                                collection: config.collection.clone(),
                            };
                            
                            self.event_sender.send(file_event).await?;
                        }
                    }
                }
                _ => {}
            }
        }
        
        Ok(())
    }
    
    fn should_process_file(&self, path: &Path, config: &WatchConfig) -> bool {
        // Check file patterns
        if !config.file_patterns.is_empty() {
            let matches_pattern = config.file_patterns.iter().any(|pattern| {
                glob_match::glob_match(pattern, &path.to_string_lossy())
            });
            if !matches_pattern {
                return false;
            }
        }
        
        // Check ignore patterns
        for ignore_pattern in &config.ignore_patterns {
            if glob_match::glob_match(ignore_pattern, &path.to_string_lossy()) {
                return false;
            }
        }
        
        true
    }
}
```

### Event Debouncing

```rust
// src/watching/debouncer.rs
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokio::time::timeout;

pub struct EventDebouncer {
    pending_events: HashMap<PathBuf, PendingEvent>,
    debounce_duration: Duration,
}

struct PendingEvent {
    event: Event,
    last_update: Instant,
}

impl EventDebouncer {
    pub fn new(debounce_duration: Duration) -> Self {
        Self {
            pending_events: HashMap::new(),
            debounce_duration,
        }
    }
    
    pub async fn process_event(&mut self, event: Event) -> Result<Option<Event>, WatchError> {
        let paths = event.paths.clone();
        
        for path in paths {
            let now = Instant::now();
            
            match self.pending_events.get_mut(&path) {
                Some(pending) => {
                    // Update existing pending event
                    pending.event = event.clone();
                    pending.last_update = now;
                }
                None => {
                    // Add new pending event
                    self.pending_events.insert(path.clone(), PendingEvent {
                        event: event.clone(),
                        last_update: now,
                    });
                }
            }
        }
        
        // Wait for debounce period
        tokio::time::sleep(self.debounce_duration).await;
        
        // Check if event is still pending and stable
        for path in event.paths {
            if let Some(pending) = self.pending_events.get(&path) {
                if now.duration_since(pending.last_update) >= self.debounce_duration {
                    // Event is stable, return it
                    let stable_event = pending.event.clone();
                    self.pending_events.remove(&path);
                    return Ok(Some(stable_event));
                }
            }
        }
        
        Ok(None)
    }
}
```

## LSP Integration

### Language Server Detection

```rust
// src/lsp/detection.rs
use std::process::Command;

pub struct LspDetector {
    detected_servers: HashMap<String, LspServerInfo>,
}

pub struct LspServerInfo {
    pub command: String,
    pub args: Vec<String>,
    pub language_id: String,
    pub file_extensions: Vec<String>,
}

impl LspDetector {
    pub async fn detect_available_servers(&mut self) -> Result<(), LspError> {
        let potential_servers = [
            ("rust-analyzer", vec!["rust-analyzer"], "rust", vec!["rs"]),
            ("ruff-lsp", vec!["ruff-lsp"], "python", vec!["py"]),
            ("typescript-language-server", vec!["typescript-language-server", "--stdio"], "typescript", vec!["ts", "tsx"]),
            ("eslint", vec!["vscode-eslint-language-server", "--stdio"], "javascript", vec!["js", "jsx"]),
            ("gopls", vec!["gopls"], "go", vec!["go"]),
            ("clangd", vec!["clangd"], "cpp", vec!["cpp", "cc", "cxx", "c", "h", "hpp"]),
        ];
        
        for (name, command_parts, lang, extensions) in potential_servers {
            if self.is_command_available(&command_parts[0]).await? {
                self.detected_servers.insert(name.to_string(), LspServerInfo {
                    command: command_parts[0].to_string(),
                    args: command_parts[1..].to_vec(),
                    language_id: lang.to_string(),
                    file_extensions: extensions.into_iter().map(String::from).collect(),
                });
                
                info!("Detected LSP server: {} for language: {}", name, lang);
            }
        }
        
        Ok(())
    }
    
    pub fn get_server_for_file(&self, file_path: &Path) -> Option<&LspServerInfo> {
        let extension = file_path.extension()?.to_str()?;
        
        self.detected_servers.values()
            .find(|server| server.file_extensions.contains(&extension.to_string()))
    }
    
    async fn is_command_available(&self, command: &str) -> Result<bool, LspError> {
        let output = Command::new("which")
            .arg(command)
            .output()
            .await?;
        
        Ok(output.status.success())
    }
}
```

### LSP Client Implementation

```rust
// src/lsp/client.rs
use tower_lsp::jsonrpc::{Request, Response};
use tower_lsp::lsp_types::*;

pub struct LspClient {
    server_info: LspServerInfo,
    process: Option<tokio::process::Child>,
    request_id: u64,
}

impl LspClient {
    pub async fn start(&mut self, workspace_root: &Path) -> Result<(), LspError> {
        // Start LSP server process
        let mut command = tokio::process::Command::new(&self.server_info.command);
        command.args(&self.server_info.args);
        command.stdin(Stdio::piped());
        command.stdout(Stdio::piped());
        command.stderr(Stdio::piped());
        
        let mut process = command.spawn()?;
        
        // Initialize LSP connection
        let init_request = InitializeParams {
            process_id: Some(std::process::id()),
            root_path: Some(workspace_root.to_string_lossy().to_string()),
            root_uri: Some(Url::from_file_path(workspace_root).unwrap()),
            initialization_options: None,
            capabilities: ClientCapabilities {
                text_document: Some(TextDocumentClientCapabilities {
                    hover: Some(HoverClientCapabilities {
                        dynamic_registration: Some(false),
                        content_format: Some(vec![MarkupKind::Markdown]),
                    }),
                    definition: Some(GotoCapability {
                        dynamic_registration: Some(false),
                        link_support: Some(false),
                    }),
                    document_symbol: Some(DocumentSymbolClientCapabilities {
                        dynamic_registration: Some(false),
                        symbol_kind: None,
                        hierarchical_document_symbol_support: Some(true),
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            },
            ..Default::default()
        };
        
        self.send_request("initialize", init_request).await?;
        self.send_notification("initialized", InitializedParams {}).await?;
        
        self.process = Some(process);
        Ok(())
    }
    
    pub async fn get_document_symbols(&mut self, file_path: &Path) -> Result<Vec<SymbolInformation>, LspError> {
        let uri = Url::from_file_path(file_path).unwrap();
        let content = tokio::fs::read_to_string(file_path).await?;
        
        // Open document
        self.send_notification("textDocument/didOpen", DidOpenTextDocumentParams {
            text_document: TextDocumentItem {
                uri: uri.clone(),
                language_id: self.server_info.language_id.clone(),
                version: 1,
                text: content,
            },
        }).await?;
        
        // Request symbols
        let params = DocumentSymbolParams {
            text_document: TextDocumentIdentifier { uri: uri.clone() },
            work_done_progress_params: Default::default(),
            partial_result_params: Default::default(),
        };
        
        let response: Option<DocumentSymbolResponse> = self
            .send_request("textDocument/documentSymbol", params)
            .await?;
        
        let symbols = match response {
            Some(DocumentSymbolResponse::Flat(symbols)) => symbols,
            Some(DocumentSymbolResponse::Nested(symbols)) => {
                // Convert nested symbols to flat representation
                self.flatten_document_symbols(symbols)
            }
            None => vec![],
        };
        
        // Close document
        self.send_notification("textDocument/didClose", DidCloseTextDocumentParams {
            text_document: TextDocumentIdentifier { uri },
        }).await?;
        
        Ok(symbols)
    }
}
```

## Engine Lifecycle Management

### Graceful Startup and Shutdown

```rust
// src/main.rs
use tokio::signal;
use tracing::{info, warn, error};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::init();
    
    // Parse configuration
    let config = Config::from_env_and_args()?;
    
    // Initialize engine components
    let mut engine = IngestionEngine::new(config).await?;
    
    // Start gRPC server
    let grpc_task = tokio::spawn(async move {
        engine.start_grpc_server().await
    });
    
    // Setup graceful shutdown
    let shutdown_signal = setup_shutdown_handlers().await;
    
    info!("Rust ingestion engine started");
    
    // Wait for shutdown signal or gRPC server to exit
    tokio::select! {
        result = grpc_task => {
            match result {
                Ok(Ok(())) => info!("gRPC server exited successfully"),
                Ok(Err(e)) => error!("gRPC server error: {}", e),
                Err(e) => error!("gRPC task panic: {}", e),
            }
        }
        _ = shutdown_signal => {
            info!("Shutdown signal received");
        }
    }
    
    // Graceful shutdown sequence
    info!("Starting graceful shutdown...");
    
    // 1. Stop accepting new requests
    engine.stop_accepting_requests().await;
    
    // 2. Wait for current tasks to complete (with timeout)
    let shutdown_timeout = Duration::from_secs(30);
    match tokio::time::timeout(shutdown_timeout, engine.wait_for_tasks()).await {
        Ok(()) => info!("All tasks completed successfully"),
        Err(_) => {
            warn!("Shutdown timeout reached, forcing shutdown");
            engine.force_shutdown().await;
        }
    }
    
    // 3. Cleanup resources
    engine.cleanup().await?;
    
    info!("Rust ingestion engine stopped");
    Ok(())
}

async fn setup_shutdown_handlers() -> impl std::future::Future<Output = ()> {
    let ctrl_c = async {
        signal::ctrl_c().await
            .expect("Failed to install Ctrl+C handler");
    };
    
    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install signal handler")
            .recv()
            .await;
    };
    
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    
    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
```

### Engine State Management

```rust
// src/lib.rs (for Python bindings)
use pyo3::prelude::*;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone)]
pub enum EngineState {
    Stopped,
    Starting,
    Running,
    Stopping,
    Error(String),
}

#[pyclass]
pub struct RustIngestionEngine {
    state: Arc<RwLock<EngineState>>,
    config: EngineConfig,
    runtime: Option<tokio::runtime::Runtime>,
    grpc_port: u16,
    task_queue: Arc<TaskQueue>,
}

#[pymethods]
impl RustIngestionEngine {
    #[new]
    fn new(config_dict: std::collections::HashMap<String, PyObject>) -> PyResult<Self> {
        let config = EngineConfig::from_python_dict(config_dict)?;
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        
        Ok(Self {
            state: Arc::new(RwLock::new(EngineState::Stopped)),
            config,
            runtime: Some(runtime),
            grpc_port: 0,
            task_queue: Arc::new(TaskQueue::new()),
        })
    }
    
    fn start(&mut self, py: Python<'_>) -> PyResult<()> {
        let state = self.state.clone();
        let config = self.config.clone();
        let task_queue = self.task_queue.clone();
        
        py.allow_threads(|| {
            if let Some(runtime) = &self.runtime {
                runtime.block_on(async {
                    *state.write().await = EngineState::Starting;
                    
                    match self.start_internal(config, task_queue).await {
                        Ok(port) => {
                            self.grpc_port = port;
                            *state.write().await = EngineState::Running;
                        }
                        Err(e) => {
                            *state.write().await = EngineState::Error(e.to_string());
                            return Err(e);
                        }
                    }
                    
                    Ok(())
                })
            } else {
                Err(PyRuntimeError::new_err("Runtime not available"))
            }
        })
    }
    
    fn stop(&mut self, py: Python<'_>) -> PyResult<()> {
        let state = self.state.clone();
        let task_queue = self.task_queue.clone();
        
        py.allow_threads(|| {
            if let Some(runtime) = &self.runtime {
                runtime.block_on(async {
                    *state.write().await = EngineState::Stopping;
                    
                    // Graceful shutdown: wait for current tasks
                    task_queue.stop_accepting_new_tasks().await;
                    
                    // Wait for tasks with timeout
                    let timeout = Duration::from_secs(30);
                    match tokio::time::timeout(timeout, task_queue.wait_for_completion()).await {
                        Ok(()) => {
                            *state.write().await = EngineState::Stopped;
                        }
                        Err(_) => {
                            // Force shutdown after timeout
                            task_queue.force_shutdown().await;
                            *state.write().await = EngineState::Stopped;
                        }
                    }
                })
            } else {
                Ok(())
            }
        })
    }
    
    fn grpc_port(&self) -> u16 {
        self.grpc_port
    }
    
    fn get_state(&self, py: Python<'_>) -> PyResult<String> {
        py.allow_threads(|| {
            if let Some(runtime) = &self.runtime {
                runtime.block_on(async {
                    let state = self.state.read().await;
                    Ok(format!("{:?}", *state))
                })
            } else {
                Ok("Runtime not available".to_string())
            }
        })
    }
}
```

## gRPC Protocol Definition

### Core Protocol Buffer Definitions

```protobuf
// proto/ingestion.proto
syntax = "proto3";

package workspace_qdrant_mcp.ingestion;

// Core ingestion service
service IngestionEngine {
    // Engine lifecycle
    rpc StartEngine(StartEngineRequest) returns (EngineStatus);
    rpc GetEngineStatus(Empty) returns (EngineStatus);
    rpc StopEngine(StopEngineRequest) returns (EngineStatus);
    
    // Document processing
    rpc IngestDocument(IngestDocumentRequest) returns (stream IngestProgress);
    rpc IngestFolder(IngestFolderRequest) returns (stream IngestProgress);
    rpc ProcessYamlMetadata(ProcessYamlRequest) returns (stream IngestProgress);
    
    // File watching
    rpc StartWatching(WatchRequest) returns (WatchStatus);
    rpc StopWatching(StopWatchRequest) returns (WatchStatus);
    rpc GetWatchStatus(Empty) returns (WatchStatus);
    
    // LSP operations
    rpc GetAvailableLsps(Empty) returns (LspServerList);
    rpc AnalyzeCodeFile(CodeAnalysisRequest) returns (CodeAnalysisResponse);
}

// Basic types
message Empty {}

// Engine lifecycle messages
message StartEngineRequest {
    map<string, string> config = 1;
}

message StopEngineRequest {
    bool force_shutdown = 1;
    int32 timeout_seconds = 2;
}

message EngineStatus {
    enum Status {
        STOPPED = 0;
        STARTING = 1;
        RUNNING = 2;
        STOPPING = 3;
        ERROR = 4;
    }
    
    Status status = 1;
    string message = 2;
    int32 active_tasks = 3;
    int32 queued_tasks = 4;
    int64 uptime_seconds = 5;
    ResourceUsage resource_usage = 6;
}

message ResourceUsage {
    double cpu_usage = 1;
    int64 memory_usage_bytes = 2;
    int32 open_files = 3;
    int32 grpc_connections = 4;
}

// Document processing messages
message IngestDocumentRequest {
    string file_path = 1;
    string collection = 2;
    map<string, string> metadata = 3;
    ProcessingOptions options = 4;
}

message IngestFolderRequest {
    string folder_path = 1;
    string collection = 2;
    repeated string file_patterns = 3;
    repeated string ignore_patterns = 4;
    bool recursive = 5;
    ProcessingOptions options = 6;
}

message ProcessYamlRequest {
    string yaml_path = 1;
    string collection = 2;
    ProcessingOptions options = 3;
}

message ProcessingOptions {
    bool enable_lsp = 1;
    bool force_reprocess = 2;
    int32 chunk_size = 3;
    double chunk_overlap = 4;
    EmbeddingConfig embedding_config = 5;
}

message EmbeddingConfig {
    string model = 1;
    int32 dimensions = 2;
    map<string, string> parameters = 3;
}

message IngestProgress {
    enum Stage {
        STARTED = 0;
        ANALYZING = 1;
        EXTRACTING = 2;
        LSP_ANALYZING = 3;
        CHUNKING = 4;
        EMBEDDING = 5;
        STORING = 6;
        COMPLETED = 7;
        FAILED = 8;
    }
    
    Stage stage = 1;
    double progress_percent = 2;
    string message = 3;
    string current_file = 4;
    DocumentResult result = 5;
    string error = 6;
}

message DocumentResult {
    string document_id = 1;
    string collection = 2;
    int32 chunks_created = 3;
    int64 processing_time_ms = 4;
    DocumentMetadata metadata = 5;
}

// File watching messages
message WatchRequest {
    string path = 1;
    string collection = 2;
    bool auto_ingest = 3;
    repeated string file_patterns = 4;
    repeated string ignore_patterns = 5;
    ProcessingOptions default_options = 6;
}

message StopWatchRequest {
    string path = 1;
}

message WatchStatus {
    repeated WatchedPath watched_paths = 1;
    int32 pending_events = 2;
    int64 total_events_processed = 3;
}

message WatchedPath {
    string path = 1;
    string collection = 2;
    bool auto_ingest = 3;
    int64 events_processed = 4;
    int64 last_event_time = 5;
}

// LSP integration messages
message LspServerList {
    repeated LspServerInfo servers = 1;
}

message LspServerInfo {
    string name = 1;
    string language = 2;
    repeated string file_extensions = 3;
    string command = 4;
    repeated string args = 5;
    bool available = 6;
}

message CodeAnalysisRequest {
    string file_path = 1;
    string content = 2;
    string language = 3;
}

message CodeAnalysisResponse {
    repeated SymbolInfo symbols = 1;
    repeated ImportInfo imports = 2;
    repeated DiagnosticInfo diagnostics = 3;
    string enhanced_content = 4;
}

// Metadata and document structure
message DocumentMetadata {
    string title = 1;
    string author = 2;
    string language = 3;
    int64 file_size = 4;
    string content_type = 5;
    int64 created_time = 6;
    int64 modified_time = 7;
    map<string, string> custom_metadata = 8;
    
    // Code-specific metadata
    repeated SymbolInfo symbols = 9;
    repeated ImportInfo imports = 10;
}

message SymbolInfo {
    string name = 1;
    string kind = 2;  // function, class, variable, etc.
    string signature = 3;
    string documentation = 4;
    Location location = 5;
    repeated SymbolInfo children = 6;
}

message ImportInfo {
    string module = 1;
    repeated string imported_names = 2;
    string alias = 3;
    Location location = 4;
}

message DiagnosticInfo {
    string message = 1;
    string severity = 2;  // error, warning, info
    Location location = 3;
    string code = 4;
}

message Location {
    int32 start_line = 1;
    int32 start_column = 2;
    int32 end_line = 3;
    int32 end_column = 4;
}
```

## Integration Points with Python MCP

### Python MCP Engine Manager

```python
# src/workspace_qdrant_mcp/rust_engine/manager.py
"""
Integration manager for the Rust ingestion engine.

This module handles the lifecycle and communication between the Python MCP server
and the embedded Rust ingestion engine.
"""

import asyncio
import logging
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional, Any

from .engine import RustIngestionEngine
from .grpc_client import IngestionEngineClient
from .config import EngineConfig
from ..core.config import Config

logger = logging.getLogger(__name__)

class RustEngineIntegration:
    """Integration layer between Python MCP and Rust engine."""
    
    def __init__(self, mcp_config: Config):
        self.mcp_config = mcp_config
        self.engine_config = EngineConfig.from_mcp_config(mcp_config)
        self.engine: Optional[RustIngestionEngine] = None
        self.grpc_client: Optional[IngestionEngineClient] = None
        self._startup_lock = asyncio.Lock()
        
    async def ensure_engine_started(self) -> bool:
        """Ensure the Rust engine is running and ready."""
        async with self._startup_lock:
            if self.engine and self.grpc_client:
                try:
                    status = await self.grpc_client.get_engine_status()
                    if status.status == "RUNNING":
                        return True
                except Exception as e:
                    logger.warning(f"Engine status check failed: {e}")
                    await self._cleanup_engine()
            
            return await self._start_engine()
    
    async def _start_engine(self) -> bool:
        """Start the embedded Rust engine."""
        try:
            # Initialize embedded engine
            self.engine = RustIngestionEngine(self.engine_config.to_dict())
            await self.engine.start()
            
            # Create gRPC client
            self.grpc_client = IngestionEngineClient(
                f"127.0.0.1:{self.engine.grpc_port()}"
            )
            await self.grpc_client.connect()
            
            logger.info(f"Rust engine started on port {self.engine.grpc_port()}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Rust engine: {e}")
            await self._cleanup_engine()
            return False
    
    async def ingest_document(
        self,
        file_path: str,
        collection: str,
        metadata: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Ingest a document through the Rust engine."""
        if not await self.ensure_engine_started():
            raise RuntimeError("Failed to start Rust engine")
        
        request = {
            "file_path": file_path,
            "collection": collection,
            "metadata": metadata or {},
            "options": options or {}
        }
        
        async for progress in self.grpc_client.ingest_document(request):
            yield {
                "stage": progress.stage,
                "progress": progress.progress_percent,
                "message": progress.message,
                "current_file": progress.current_file,
                "error": progress.error if progress.error else None
            }
    
    async def start_watching(
        self,
        path: str,
        collection: str,
        auto_ingest: bool = True,
        file_patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Start watching a path for file changes."""
        if not await self.ensure_engine_started():
            raise RuntimeError("Failed to start Rust engine")
        
        request = {
            "path": path,
            "collection": collection,
            "auto_ingest": auto_ingest,
            "file_patterns": file_patterns or [],
            "ignore_patterns": ignore_patterns or [".git/*", "node_modules/*", "__pycache__/*"]
        }
        
        return await self.grpc_client.start_watching(request)
    
    async def get_available_lsps(self) -> List[Dict[str, Any]]:
        """Get list of available LSP servers."""
        if not await self.ensure_engine_started():
            return []
        
        return await self.grpc_client.get_available_lsps()
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the Rust engine."""
        if self.engine:
            try:
                await self.engine.stop()
                logger.info("Rust engine stopped gracefully")
            except Exception as e:
                logger.error(f"Error stopping Rust engine: {e}")
        
        await self._cleanup_engine()
    
    async def _cleanup_engine(self) -> None:
        """Clean up engine resources."""
        if self.grpc_client:
            await self.grpc_client.close()
            self.grpc_client = None
        
        self.engine = None

# Global engine manager instance
_engine_manager: Optional[RustEngineIntegration] = None

def get_engine_manager(config: Config) -> RustEngineIntegration:
    """Get or create the global engine manager instance."""
    global _engine_manager
    
    if _engine_manager is None:
        _engine_manager = RustEngineIntegration(config)
    
    return _engine_manager
```

### Enhanced MCP Tools with Rust Engine

```python
# src/workspace_qdrant_mcp/tools/rust_enhanced_tools.py
"""
MCP tools enhanced with Rust engine processing capabilities.

These tools provide the same interface as existing MCP tools but leverage
the Rust engine for improved performance and additional capabilities.
"""

from typing import Dict, List, Optional, Any, AsyncIterator
from fastmcp import Context

from ..rust_engine.manager import get_engine_manager
from ..core.config import Config

async def ingest_document_rust(
    ctx: Context,
    file_path: str,
    collection: str,
    metadata: Optional[Dict[str, str]] = None,
    enable_lsp: bool = True,
    chunk_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Ingest a document using the Rust processing engine.
    
    This tool provides the same functionality as the existing add_document tool
    but uses the Rust engine for improved performance and enhanced processing.
    
    Args:
        file_path: Path to the document to ingest
        collection: Target collection name (validates against reserved names)
        metadata: Optional metadata dictionary
        enable_lsp: Whether to use LSP analysis for code files
        chunk_size: Custom chunk size (uses intelligent defaults if not specified)
    
    Returns:
        Dictionary containing ingestion results and statistics
    """
    config = Config()
    engine_manager = get_engine_manager(config)
    
    # Validate collection name against reserved naming conventions
    if collection.startswith('_'):
        if collection == 'memory':
            pass  # memory collection allowed
        elif not collection.startswith('_'):
            return {
                "error": f"Collection '{collection}' conflicts with reserved library naming pattern"
            }
    
    processing_options = {
        "enable_lsp": enable_lsp,
        "chunk_size": chunk_size or config.chunk_size
    }
    
    final_result = {}
    progress_messages = []
    
    try:
        async for progress in engine_manager.ingest_document(
            file_path=file_path,
            collection=collection,
            metadata=metadata,
            options=processing_options
        ):
            progress_messages.append(progress["message"])
            
            if progress.get("error"):
                return {
                    "success": False,
                    "error": progress["error"],
                    "progress": progress_messages
                }
            
            if progress["stage"] == "COMPLETED":
                final_result = progress
                break
    
        return {
            "success": True,
            "document_id": final_result.get("result", {}).get("document_id"),
            "chunks_created": final_result.get("result", {}).get("chunks_created", 0),
            "processing_time_ms": final_result.get("result", {}).get("processing_time_ms", 0),
            "collection": collection,
            "progress": progress_messages
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "progress": progress_messages
        }

async def start_library_watching(
    ctx: Context,
    library_path: str,
    library_name: str,
    file_patterns: Optional[List[str]] = None,
    auto_ingest: bool = True
) -> Dict[str, Any]:
    """
    Start watching a library folder for automatic ingestion.
    
    This enables the v2.0 library watching capability where users can configure
    folders to be automatically ingested into library collections.
    
    Args:
        library_path: Path to the library folder to watch
        library_name: Name of the library (will be prefixed with _)
        file_patterns: File patterns to watch (e.g., ["*.pdf", "*.epub"])
        auto_ingest: Whether to automatically ingest changes
    
    Returns:
        Dictionary with watch configuration and status
    """
    config = Config()
    engine_manager = get_engine_manager(config)
    
    # Enforce library naming convention
    collection_name = f"_{library_name}"
    
    # Default patterns for common document types
    if file_patterns is None:
        file_patterns = [
            "*.pdf", "*.epub", "*.mobi", "*.txt", "*.md", "*.rst",
            "*.py", "*.rs", "*.js", "*.ts", "*.go", "*.cpp", "*.c", "*.h"
        ]
    
    try:
        result = await engine_manager.start_watching(
            path=library_path,
            collection=collection_name,
            auto_ingest=auto_ingest,
            file_patterns=file_patterns,
            ignore_patterns=[
                ".git/*", ".svn/*", "node_modules/*", "__pycache__/*",
                "target/*", "dist/*", "build/*", ".venv/*", "venv/*"
            ]
        )
        
        return {
            "success": True,
            "library_name": library_name,
            "collection": collection_name,
            "watched_path": library_path,
            "file_patterns": file_patterns,
            "auto_ingest": auto_ingest,
            "watch_status": result
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "library_name": library_name,
            "collection": collection_name
        }

async def get_engine_status(ctx: Context) -> Dict[str, Any]:
    """
    Get comprehensive status of the Rust ingestion engine.
    
    Returns detailed information about engine state, resource usage,
    active tasks, and LSP server availability.
    
    Returns:
        Dictionary containing engine status and diagnostics
    """
    config = Config()
    engine_manager = get_engine_manager(config)
    
    try:
        if not engine_manager.engine or not engine_manager.grpc_client:
            return {
                "engine_status": "STOPPED",
                "message": "Engine not started",
                "grpc_connection": False
            }
        
        status = await engine_manager.grpc_client.get_engine_status()
        lsp_servers = await engine_manager.get_available_lsps()
        
        return {
            "engine_status": status.status,
            "message": status.message,
            "active_tasks": status.active_tasks,
            "queued_tasks": status.queued_tasks,
            "uptime_seconds": status.uptime_seconds,
            "resource_usage": {
                "cpu_usage": status.resource_usage.cpu_usage,
                "memory_usage_mb": status.resource_usage.memory_usage_bytes / (1024 * 1024),
                "open_files": status.resource_usage.open_files,
                "grpc_connections": status.resource_usage.grpc_connections
            },
            "lsp_servers": [
                {
                    "name": lsp.name,
                    "language": lsp.language,
                    "available": lsp.available,
                    "extensions": lsp.file_extensions
                } for lsp in lsp_servers
            ],
            "grpc_connection": True,
            "grpc_port": engine_manager.engine.grpc_port()
        }
        
    except Exception as e:
        return {
            "engine_status": "ERROR",
            "error": str(e),
            "grpc_connection": False
        }
```

## Platform Distribution Strategy

### Build Configuration

The Rust engine is distributed as part of the Python package using pre-built wheels for major platforms:

**Tier 1 Platforms** (Pre-built wheels):
- macOS Intel (x86_64-apple-darwin)
- macOS Apple Silicon (aarch64-apple-darwin)
- Linux x86_64 (x86_64-unknown-linux-gnu)
- Windows x64 (x86_64-pc-windows-msvc)
- Windows ARM64 (aarch64-pc-windows-msvc)

**Tier 2 Platforms** (Source distribution):
- Linux ARM64 (aarch64-unknown-linux-gnu)
- Other Linux variants (musl, etc.)

### PyPI Package Structure

```
workspace-qdrant-mcp-2.0.0/
├── src/
│   └── workspace_qdrant_mcp/
│       ├── rust_engine/
│       │   ├── __init__.py
│       │   ├── _rust_engine.cpython-310-darwin.so  # macOS wheel
│       │   ├── _rust_engine.cpython-310-win_amd64.pyd  # Windows wheel
│       │   └── _rust_engine.cpython-310-linux-x86_64.so  # Linux wheel
│       └── ...
├── rust-engine/  # Rust source (for source builds)
├── pyproject.toml
└── README.md
```

### Installation Experience

For users on Tier 1 platforms:
```bash
pip install workspace-qdrant-mcp==2.0.0
# Downloads pre-built wheel with embedded Rust engine (~15-25MB)
# Ready to use immediately
```

For users on Tier 2 platforms:
```bash
pip install workspace-qdrant-mcp==2.0.0
# Downloads source distribution
# Compiles Rust engine during installation (requires Rust toolchain)
# Installation time: 2-5 minutes depending on hardware
```

## Success Criteria and Validation

### Architecture Validation Checklist

- [ ] **Embedded Deployment**: Rust engine successfully builds as Python extension
- [ ] **gRPC Communication**: Python MCP ↔ Rust engine communication established
- [ ] **Graceful Lifecycle**: Engine starts on MCP load, stops gracefully on shutdown
- [ ] **Document Processing**: All file formats (text, PDF, EPUB, code) processed correctly
- [ ] **LSP Integration**: Language servers detected and integrated for code analysis
- [ ] **File Watching**: Cross-platform file watching with debouncing implemented
- [ ] **Collection Naming**: Reserved collection naming enforced (`memory`, `_library`, project-*)
- [ ] **Platform Distribution**: Tier 1 platforms build successfully in CI/CD
- [ ] **Performance Targets**: Processing throughput meets or exceeds current Python implementation
- [ ] **Integration Points**: Python MCP tools successfully integrate with Rust engine

### Performance Benchmarks

**Target Performance Metrics**:
- Document ingestion: 1000+ documents/minute (10x improvement over Python)
- Search response time: <100ms (maintained from current implementation)
- Memory usage: <500MB for 100k+ documents
- Engine startup time: <2 seconds
- Graceful shutdown time: <30 seconds with work completion

**Integration Success Metrics**:
- Zero-downtime engine restarts
- 100% compatibility with existing MCP tools
- Successful compilation on all Tier 1 platforms
- LSP server detection accuracy >90% for common languages

## Implementation Timeline

### Phase 1: Core Engine Foundation (Week 1-2)
- [ ] Rust project structure and dependencies
- [ ] gRPC protocol definition and basic server
- [ ] Python embedding with maturin configuration
- [ ] Basic document processing pipeline (text files)
- [ ] Engine lifecycle management

### Phase 2: Enhanced Processing (Week 2-3)  
- [ ] PDF and EPUB processing
- [ ] Code file processing with Tree-sitter
- [ ] LSP integration and server detection
- [ ] File watching system
- [ ] Embedding generation integration

### Phase 3: Python Integration (Week 3-4)
- [ ] Python MCP integration layer
- [ ] Enhanced MCP tools using Rust engine
- [ ] Collection naming validation
- [ ] Error handling and recovery
- [ ] Comprehensive testing

### Phase 4: Platform Distribution (Week 4)
- [ ] CI/CD pipeline for multi-platform builds
- [ ] Wheel generation and testing
- [ ] Documentation and deployment guides
- [ ] Performance validation and benchmarking

This comprehensive architecture provides the foundation for the entire v2.0 architecture track, enabling memory-driven behavior, library management, version-aware document handling, and high-performance processing through the embedded Rust engine.