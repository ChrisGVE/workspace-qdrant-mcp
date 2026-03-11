## Document Type Taxonomy

Library documents are categorized into two families based on their layout characteristics, each requiring different processing pipelines.

### Document Families

#### Page-Based Documents (Fixed Layout)

Documents with explicit page boundaries and fixed positions. Processing extracts text page-by-page with page numbers as locators.

**Supported Formats:**

| Format | Extension | Extractor | Notes |
|--------|-----------|-----------|-------|
| PDF | `.pdf` | `pdf-extract` → `lopdf` fallback | Wrapped in `catch_unwind` to trap Type 1 font encoding panics. If `pdf-extract` panics or errors, `lopdf` is tried. If neither yields text (e.g. old scanned documents with no text layer), the file is treated as image-only: indexed by filename/metadata, not an error. |
| Microsoft Word | `.docx` | ZIP + XML | Extracts from `word/document.xml` |
| Microsoft Word (Legacy) | `.doc` | Text extraction | Limited support, fallback to text mode |
| Microsoft PowerPoint | `.pptx` | ZIP + XML | Extracts slide text from `ppt/slides/slide*.xml` |
| Microsoft PowerPoint (Legacy) | `.ppt` | Text extraction | Limited support, fallback to text mode |
| OpenDocument Text | `.odt` | ZIP + XML | Extracts from `content.xml` |
| OpenDocument Presentation | `.odp` | ZIP + XML | Extracts from `content.xml` |
| OpenDocument Spreadsheet | `.ods` | ZIP + XML | Extracts from `content.xml` |
| Rich Text Format | `.rtf` | RTF parser | Strips RTF control codes |

#### Stream-Based Documents (Flowing Text)

Documents with continuous, flowing text where content order is primary. Processing extracts text as a stream with chapter/section markers as locators.

**Supported Formats:**

| Format | Extension | Extractor | Notes |
|--------|-----------|-----------|-------|
| EPUB | `.epub` | `epub` crate | Extracts from all chapters, converts HTML to text |
| HTML | `.html`, `.htm` | `html2text` | Converts HTML to plain text |
| Markdown | `.md`, `.markdown` | UTF-8 text | Native text format with frontmatter support |
| Plain Text | `.txt` | UTF-8 with encoding detection | Uses `chardet` for non-UTF-8 files |

### Title Extraction

Document titles are extracted using a three-level priority cascade to ensure meaningful titles for search results.

**Priority Cascade:**

1. **Embedded Metadata** (highest priority)
   - PDF: Info dictionary `/Title` field (via `lopdf`)
   - DOCX/PPTX: `docProps/core.xml` `dc:title` element
   - EPUB: OPF metadata (extracted by `epub` crate)
   - ODT/ODP/ODS: `meta.xml` `dc:title` element
   - RTF: `{\info{\title ...}}` block
   - HTML: `<title>` tag or `og:title` meta tag

2. **Content Heuristics** (fallback)
   - HTML: First `<h1>` element (with tags stripped)
   - Markdown: YAML frontmatter `title:` field or first `# Heading`
   - Plain text: First non-empty line (if ≤200 chars, no trailing punctuation, contains uppercase)

3. **Filename Fallback** (last resort)
   - Convert filename stem to title case
   - Replace underscores and hyphens with spaces
   - Example: `2024_annual_report.pdf` → "2024 Annual Report"

**Placeholder Detection:**

The system detects and rejects common auto-generated titles:
- "Untitled", "Document", "Presentation", "Slide", "Book"
- Numbered placeholders: "Document1", "Slide 3"
- Microsoft Word headers: "Microsoft Word - filename.docx"

When a placeholder is detected, the cascade falls back to the next priority level.

### Extraction Architecture

**DocumentExtractor Trait (Page-Based):**

```rust
trait DocumentExtractor {
    fn extract(&self, file_path: &Path) -> Result<(String, HashMap<String, String>)>;
}

// Implementations:
- PdfExtractor (via pdf-extract)
- DocxExtractor (ZIP + XML parsing)
- PptxExtractor (ZIP + XML parsing)
- OdtExtractor (ZIP + XML parsing)
- RtfExtractor (RTF control code parser)
```

**StreamDocumentExtractor Trait (Stream-Based):**

```rust
trait StreamDocumentExtractor {
    fn extract(&self, file_path: &Path) -> Result<(String, HashMap<String, String>)>;
}

// Implementations:
- EpubExtractor (via epub crate)
- HtmlExtractor (via html2text)
- MarkdownExtractor (native UTF-8)
- TextExtractor (UTF-8 with chardet fallback)
```

**Common Metadata Fields:**

Both extractors return `HashMap<String, String>` with format-specific metadata:

| Field | Page-Based Example | Stream-Based Example |
|-------|-------------------|---------------------|
| `source_format` | `"pdf"`, `"docx"` | `"epub"`, `"markdown"` |
| `title` | Extracted via cascade | Extracted via cascade |
| `author` | From embedded metadata | From embedded metadata |
| `page_count` | Total pages (PDF, DOCX) | N/A |
| `chapter_count` | N/A | Total chapters (EPUB) |
| `images_detected` | Count (metadata only) | Count (metadata only) |

### Token-Based Chunking

Library documents use **token-based chunking** instead of character-based chunking to align with embedding model token limits and improve chunk quality.

**Configuration:**

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `chunk_target_tokens` | 105 | 90-120 | Target tokens per chunk |
| `chunk_overlap_tokens` | 12 | ~10-15% | Overlap between chunks |

**Rationale:**
- Embedding models have token-based limits (not character limits)
- 105 tokens ≈ 384 characters (previous character-based default)
- Overlap ensures context continuity across chunk boundaries

**Implementation:**

```rust
use tokenizers::Tokenizer;

fn chunk_text_by_tokens(
    text: &str,
    tokenizer: &Tokenizer,
    config: &ChunkingConfig
) -> Vec<TextChunk> {
    let encoding = tokenizer.encode(text, false).unwrap();
    let tokens = encoding.get_ids();

    let mut chunks = Vec::new();
    let mut start_token = 0;

    while start_token < tokens.len() {
        let end_token = (start_token + config.chunk_target_tokens).min(tokens.len());
        let chunk_tokens = &tokens[start_token..end_token];

        // Decode tokens back to text
        let chunk_text = tokenizer.decode(chunk_tokens, true).unwrap();

        chunks.push(TextChunk {
            content: chunk_text,
            chunk_index: chunks.len(),
            // Additional metadata...
        });

        // Advance with overlap
        start_token = end_token.saturating_sub(config.chunk_overlap_tokens);
    }

    chunks
}
```

**Chunk Payload Fields:**

| Field | Description |
|-------|-------------|
| `chunk_text_raw` | Original chunk text (no modifications) |
| `chunk_text_indexed` | Chunk with heading context prepended |
| `char_start` | Character offset in parent `unit_text` |
| `char_end` | End character offset |
| `chunk_index` | Sequence number within document |

**Heading Context Injection:**

For page-based documents, the most recent heading (from previous pages/sections) is prepended to `chunk_text_indexed` to provide hierarchical context:

```
chunk_text_raw: "The array indexing operation..."
chunk_text_indexed: "## Advanced Features\n### Array Operations\nThe array indexing operation..."
```

This ensures search results include contextual headings without duplication in the raw text.

**Rules Collection:**

```json
{
  "label": "prefer-uv", // Human-readable identifier (unique per scope)
  "content": "Use uv instead of pip for Python packages",
  "scope": "global", // global|project
  "project_id": null, // null for global, "abc123" for project-specific
  "created_at": "2026-01-30T12:00:00Z"
}
```

### Store Tool Type System

The `store` MCP tool uses a `type` parameter that describes the **nature of the input**, not the destination collection. The destination collection is determined by the type combined with context parameters.

#### Type: `library` (default)

Stores content to the `libraries` collection under a named library. Requires `content` and `libraryName` (unless `forProject` is true, in which case `libraryName` defaults to `"project-refs"`).

Libraries are collections of reference information — books, documentation, papers, websites. They are NOT programming libraries (use context7 MCP for those). This type should only be used when the user **explicitly instructs** the LLM to store into a named library — not autonomously.

#### Type: `url`

Queues a URL for daemon-side fetch and ingestion. Stores to `libraries` if `libraryName` is provided (for adding URLs to a named library), otherwise stores to `scratchpad` for ad-hoc reference. URLs can also be part of a tracked library folder as webarchive files.

#### Type: `scratchpad`

Stores content to the `scratchpad` collection. See [Scratchpad Collection](#scratchpad-collection) below.

#### Type: `project`

Registers a new project with the daemon for file watching and ingestion. Uses `register_if_new: true` so the daemon will create the project in `watch_folders` if it doesn't already exist. Can also activate and keep-active existing projects.

#### Collection Routing Summary

| Type | Destination | Condition |
|------|------------|-----------|
| `library` | `libraries` | Always (explicit user instruction only) |
| `url` | `libraries` | When `libraryName` provided |
| `url` | `scratchpad` | Default (ad-hoc reference) |
| `scratchpad` | `scratchpad` | Always |
| `project` | N/A | Registers project with daemon |

**Write restrictions:** The MCP server cannot store content to the `projects` collection — project content is ingested exclusively by the daemon via file watching. Behavioral rules use the dedicated `rules` tool.

#### Library-Type Document Routing from Projects

When the daemon's file watcher encounters library-type documents (PDF, EPUB, DOCX, PPTX, ODT, ODS, XLSX, XLS, Numbers, Parquet, RTF, Mobi, Pages, Key) inside a project folder, it routes them to the `libraries` collection with the project's tenant_id as the library name reference. This ensures reference documents within a project folder are processed by the library pipeline (structural units, token-based chunking) rather than the code pipeline.

The MCP server and CLI can mark library documents as `persistent` to prevent the daemon from deleting them when the source file is removed.

### Scratchpad Collection

The `scratchpad` collection serves as a persistent working space for ad-hoc content storage. It is the **preferred destination** for content that doesn't belong in a named library or the rules collection.

#### Scoping

Scratchpad entries can be either global or project-scoped:

| Scope | `tenant_id` | Visibility |
|-------|------------|------------|
| Global | `_global_` | Visible from all projects |
| Project-scoped | `<project_id>` | Visible only when searching from that project |

Project scoping is automatic: when a session has an active project, scratchpad entries default to that project's scope. When no project is active, entries are stored globally.

#### Organization

Global entries can be further segregated by user-defined themes or keywords via the `tags` parameter. Tags enable filtering during search (using the `tag` or `tags` search parameters).

#### Content Types

The scratchpad accepts:

- **Text strings**: Direct content from the LLM or user
- **URLs** (via `type: "url"` without `libraryName`): Daemon fetches and stores the extracted text
- **File content** (via `filePath`): Content extracted from local files
- **Notes**: Session observations, analysis transcripts, research findings

#### Use Cases

1. **Session memory**: Store analysis results, brainstorming transcripts, decision records
2. **Ad-hoc reference**: Store URLs or text snippets for later retrieval
3. **Cross-session context**: Persist information that should survive session boundaries
4. **Research collection**: Accumulate findings from multiple sources under theme tags

---

