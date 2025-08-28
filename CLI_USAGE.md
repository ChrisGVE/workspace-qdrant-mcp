# Batch Document Ingestion CLI

The `workspace-qdrant-ingest` command provides powerful batch document ingestion capabilities for the workspace-qdrant-mcp system. This CLI tool addresses the critical missing feature identified in the feature gap analysis compared to claude-qdrant-mcp.

## Quick Start

```bash
# List supported file formats
workspace-qdrant-ingest formats

# Estimate processing time for a directory
workspace-qdrant-ingest estimate /path/to/documents

# Perform batch ingestion
workspace-qdrant-ingest ingest /path/to/documents --collection my-project

# Dry run to preview operation
workspace-qdrant-ingest ingest /path/to/documents --collection my-project --dry-run
```

## Features

### 🚀 **High-Performance Batch Processing**
- Concurrent document processing with configurable limits (default: 5)
- Intelligent file discovery with recursive directory traversal
- Real-time progress tracking with rich CLI interface
- Performance metrics and processing rate reporting

### 📄 **Multi-Format Document Support**
- **Plain Text**: .txt, .text, .log, .csv, .json, .xml, .yaml, .py, .js, .html, .css, .sql, and more
- **Markdown**: .md, .markdown, .mdown with YAML frontmatter support
- **PDF Documents**: .pdf with metadata extraction and multi-page processing

### 🔍 **Advanced Content Processing**
- SHA256-based content deduplication to prevent duplicate ingestion
- Automatic encoding detection for text files
- Content cleaning and normalization
- Intelligent text chunking with overlap preservation
- Programming language detection for source code files

### ⚙️ **Flexible Configuration**
- Configurable concurrency levels for optimal performance
- Custom chunk size and overlap settings
- Format filtering to process only specific file types
- File exclusion patterns with glob support
- Dry-run mode for safe operation preview

### 📊 **Comprehensive Reporting**
- Detailed processing statistics and success rates
- Error collection and reporting with file-specific details
- Performance metrics including processing speed
- Content analysis with word counts and character counts

## Commands

### `formats` - List Supported Formats

Display all supported file formats and their parsing options:

```bash
workspace-qdrant-ingest formats
```

**Output includes:**
- File extensions supported by each parser
- Available parsing options and their defaults
- Format-specific features and capabilities

### `estimate` - Processing Time Estimation

Analyze a directory and estimate processing requirements:

```bash
workspace-qdrant-ingest estimate /path/to/documents [OPTIONS]
```

**Options:**
- `--formats TEXT`: Comma-separated list of formats to analyze
- `--concurrency INTEGER`: Concurrent processing tasks for estimation

**Output includes:**
- Number of files found by format
- Total directory size
- Estimated processing time
- File type breakdown

### `ingest` - Batch Document Ingestion

Process and ingest documents from a directory:

```bash
workspace-qdrant-ingest ingest /path/to/documents --collection COLLECTION [OPTIONS]
```

**Required Arguments:**
- `PATH`: Directory or file path to process
- `--collection, -c TEXT`: Target collection name

**Processing Options:**
- `--formats, -f TEXT`: File formats to process (e.g., "pdf,md,txt")
- `--concurrency INTEGER`: Number of concurrent processing tasks (default: 5)
- `--chunk-size INTEGER`: Maximum characters per text chunk (default: 1000)
- `--chunk-overlap INTEGER`: Character overlap between chunks (default: 200)

**Behavior Options:**
- `--dry-run`: Analyze files without actual ingestion
- `--recursive / --no-recursive`: Process subdirectories (default: recursive)
- `--exclude TEXT`: Glob patterns to exclude (can be used multiple times)
- `--progress / --no-progress`: Show progress bar (default: enabled)

**Control Options:**
- `--verbose, -v`: Enable verbose output
- `--debug`: Enable debug logging
- `--yes, -y`: Skip confirmation prompts

## Examples

### Basic Operations

```bash
# Simple ingestion of all supported formats
workspace-qdrant-ingest ingest /docs --collection my-project

# Process only PDF and Markdown files
workspace-qdrant-ingest ingest /docs -c my-project -f pdf,md

# High-concurrency processing for large datasets
workspace-qdrant-ingest ingest /docs -c my-project --concurrency 10
```

### Advanced Usage

```bash
# Exclude temporary and cache files
workspace-qdrant-ingest ingest /project \
    --collection codebase \
    --exclude "*.tmp" \
    --exclude "**/node_modules/**" \
    --exclude "**/__pycache__/**"

# Custom chunking for large documents
workspace-qdrant-ingest ingest /manuals \
    --collection documentation \
    --chunk-size 2000 \
    --chunk-overlap 300 \
    --formats pdf

# Dry run with verbose output for debugging
workspace-qdrant-ingest ingest /docs \
    --collection test \
    --dry-run \
    --verbose
```

### Workflow Examples

```bash
# 1. Check supported formats
workspace-qdrant-ingest formats

# 2. Estimate processing requirements  
workspace-qdrant-ingest estimate /large-dataset --concurrency 8

# 3. Perform dry run to validate
workspace-qdrant-ingest ingest /large-dataset \
    --collection big-project \
    --dry-run \
    --formats pdf,md,txt

# 4. Execute actual ingestion
workspace-qdrant-ingest ingest /large-dataset \
    --collection big-project \
    --concurrency 8 \
    --formats pdf,md,txt \
    --progress
```

## File Format Details

### Plain Text Files
- **Extensions**: .txt, .text, .log, .csv, .json, .xml, .yaml, .yml, .ini, .cfg, .conf, .py, .js, .html, .css, .sql, .sh, .bash, .zsh, .fish, .ps1, .bat, .cmd
- **Features**:
  - Automatic encoding detection (UTF-8, Latin-1, etc.)
  - Content cleaning and whitespace normalization
  - Programming language detection for source code
  - Text analysis and statistics generation

### Markdown Files
- **Extensions**: .md, .markdown, .mdown, .mkd, .mkdn
- **Features**:
  - YAML frontmatter extraction and parsing
  - Structure preservation (headings, lists, code blocks)
  - Link and image extraction options
  - Table of contents generation (with python-markdown)
  - Structured plain text conversion

### PDF Documents
- **Extensions**: .pdf
- **Features**:
  - Multi-page text extraction
  - PDF metadata extraction (title, author, creation date, etc.)
  - Encrypted PDF support with password
  - Page-by-page processing for large documents
  - Content analysis and statistics

## Configuration

The CLI integrates with the existing workspace-qdrant-mcp configuration system:

- **Environment Variables**: Use `.env` files for Qdrant connection settings
- **Collection Management**: Automatic collection creation and management
- **Workspace Integration**: Project-scoped operations with automatic detection

## Performance Considerations

### Concurrency Guidelines
- **Small files** (< 1MB): Use higher concurrency (8-16)
- **Large files** (> 10MB): Use lower concurrency (2-5)
- **Mixed workloads**: Start with default (5) and adjust based on performance

### Memory Usage
- Memory scales with concurrency and chunk size
- Large PDFs may require significant memory for processing
- Monitor system resources during large batch operations

### Processing Speed
- Typical rates: 50-200 files per minute depending on size and format
- PDF processing is slower than text/markdown
- Network latency to Qdrant affects ingestion speed

## Error Handling

The CLI provides comprehensive error handling:

- **File-level errors**: Continue processing other files, collect error details
- **Format errors**: Skip unsupported files with clear reporting
- **Connection errors**: Graceful failure with diagnostic information
- **Validation errors**: Pre-flight checks to prevent partial failures

## Integration with Existing System

The CLI seamlessly integrates with workspace-qdrant-mcp:

- **Collection Management**: Uses existing collection creation and management
- **Embedding Generation**: Leverages existing embedding service and models
- **Workspace Detection**: Automatic project and subproject detection
- **Configuration**: Shares configuration with MCP server
- **Metadata**: Compatible metadata format with existing document tools

## Troubleshooting

### Common Issues

**1. Collection Not Found**
```bash
# List available collections first
workspace-qdrant-admin list-collections

# Create collection if needed (use existing admin tools)
```

**2. Permission Denied**
```bash
# Check file permissions
ls -la /path/to/documents

# Ensure read access to all files in directory
```

**3. Out of Memory**
```bash
# Reduce concurrency
workspace-qdrant-ingest ingest /docs -c project --concurrency 2

# Reduce chunk size
workspace-qdrant-ingest ingest /docs -c project --chunk-size 500
```

**4. Connection Issues**
```bash
# Verify Qdrant configuration
workspace-qdrant-admin health

# Check connection settings
cat .env
```

### Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
workspace-qdrant-ingest ingest /docs -c project --debug
```

This provides:
- Detailed processing logs for each file
- Parser selection and processing details
- Timing information for performance analysis
- Full error stack traces

## Logs

The CLI generates detailed logs:

- **Console output**: Progress and summary information
- **Log file**: `workspace_qdrant_ingest.log` with detailed processing information
- **Error details**: File-specific error information with context

## Future Enhancements

Potential future improvements:
- Additional document formats (DOCX, RTF, etc.)
- Advanced filtering and selection criteria
- Resume functionality for interrupted operations  
- Incremental ingestion with change detection
- Integration with CI/CD pipelines
- Batch update and deletion operations