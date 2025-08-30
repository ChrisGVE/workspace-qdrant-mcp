# YAML Metadata Workflow Guide

The YAML metadata workflow enables library ingestion with user-provided metadata completion. This system automatically detects document types, extracts available metadata, and generates user-friendly YAML files for metadata completion.

## Overview

The workflow follows these steps:

1. **Discovery**: Engine discovers files in library folder
2. **Analysis**: Documents are analyzed for type detection and metadata extraction
3. **Generation**: YAML file is created with detected vs required metadata fields
4. **User Completion**: User fills in missing metadata in the YAML file
5. **Processing**: Engine processes files with complete metadata
6. **Iteration**: YAML is updated with remaining incomplete files until all are processed

## Supported Document Types

### Book
- **Primary Version**: `edition`
- **Required**: `title`, `author`, `edition`
- **Optional**: `isbn`, `publisher`, `year`, `language`, `pages`, `genre`, `tags`

### Scientific Article
- **Primary Version**: `publication_date`
- **Required**: `title`, `authors`, `journal`, `publication_date`
- **Optional**: `doi`, `volume`, `issue`, `pages`, `abstract`, `keywords`, `tags`

### Webpage
- **Primary Version**: `ingestion_date`
- **Required**: `title`, `url`, `ingestion_date`
- **Optional**: `author`, `site_name`, `description`, `tags`, `last_modified`

### Report
- **Primary Version**: `publication_date`
- **Required**: `title`, `author`, `publication_date`
- **Optional**: `organization`, `report_number`, `pages`, `abstract`, `tags`

### Presentation
- **Primary Version**: `date`
- **Required**: `title`, `author`, `date`
- **Optional**: `event`, `location`, `slides`, `duration`, `tags`

### Manual
- **Primary Version**: `version`
- **Required**: `title`, `version`
- **Optional**: `author`, `product`, `company`, `date`, `pages`, `tags`

### Unknown
- **Primary Version**: `date`
- **Required**: `title`
- **Optional**: `author`, `date`, `source`, `type`, `tags`

## CLI Commands

### Generate YAML Metadata File

```bash
wqm ingest generate-yaml <library-path> --collection <collection-name> [options]
```

**Required Arguments:**
- `library-path`: Path to folder containing documents
- `--collection, -c`: Target library collection name (must start with '_')

**Options:**
- `--output, -o`: Output YAML file path (default: `<library-path>/metadata_completion.yaml`)
- `--format, -f`: File formats to process (e.g., `pdf,md,txt`)
- `--force`: Overwrite existing YAML file

**Example:**
```bash
# Generate YAML for personal library
wqm ingest generate-yaml /Users/john/Documents/Library --collection _personal_library

# Generate with specific formats and output location
wqm ingest generate-yaml /Users/john/Books --collection _book_collection \\
  --format pdf,epub --output /Users/john/books_metadata.yaml --force
```

### Process YAML Metadata File

```bash
wqm ingest yaml <yaml-file> [options]
```

**Required Arguments:**
- `yaml-file`: Path to completed YAML metadata file

**Options:**
- `--dry-run`: Analyze and validate without actually processing
- `--force`: Overwrite existing documents in collection

**Example:**
```bash
# Process completed metadata
wqm ingest yaml /Users/john/Documents/Library/metadata_completion.yaml

# Dry run to validate metadata
wqm ingest yaml /Users/john/books_metadata.yaml --dry-run
```

## YAML File Structure

### Header Section
```yaml
metadata:
  generated_at: '2024-08-30T22:30:00.000Z'
  engine_version: '1.0.0'
  library_collection: '_personal_library'
  instructions:
  - Fill in the required metadata fields marked with '?'
  - You can modify detected metadata if it's incorrect
  - Remove files from pending_files if you don't want to process them
  - Run 'wqm ingest yaml <this-file>' when complete
```

### Document Type Schemas
Reference section showing available document types and their metadata requirements.

### Pending Files
```yaml
pending_files:
- path: /library/advanced_python.pdf
  document_type: book
  confidence: 0.85
  detected_metadata:
    title: Advanced Python Programming
    page_count: 450
  required_metadata:
    title: Advanced Python Programming
    author: '?'              # FILL IN
    edition: '1st'           # MODIFY IF NEEDED
    isbn: '?'                # FILL IN IF AVAILABLE
    tags:
    - python
    - programming
```

### Completed Files
Files that have been successfully processed are moved here for tracking.

## Metadata Completion Guide

### Required Fields (marked with '?')
- Must be completed before document can be processed
- Replace '?' with actual values
- Use empty string '' if value is genuinely not available

### Detected Metadata
- Pre-filled based on document analysis
- Can be modified if incorrect
- Used as suggestions for completion

### Tags
- Use YAML list format:
  ```yaml
  tags:
  - tag1
  - tag2
  - tag3
  ```
- Or single line format: `tags: [tag1, tag2, tag3]`

### Multiple Authors
For scientific articles with multiple authors:
```yaml
authors: John Smith, Jane Doe, Bob Wilson
```
Or use YAML list format:
```yaml
authors:
- John Smith
- Jane Doe
- Bob Wilson
```

### Dates
Use consistent date formats:
- Full date: `2024-08-30`
- Year only: `2024`
- Month/Year: `August 2024`

## Workflow Examples

### Example 1: Book Collection
```bash
# 1. Generate YAML for book collection
wqm ingest generate-yaml ~/Books --collection _personal_books

# 2. Edit ~/Books/metadata_completion.yaml
#    Fill in author, ISBN, publisher information

# 3. Process completed metadata
wqm ingest yaml ~/Books/metadata_completion.yaml

# 4. Repeat steps 2-3 until all books are processed
```

### Example 2: Research Papers
```bash
# 1. Generate YAML for research collection
wqm ingest generate-yaml ~/Papers --collection _research_library --format pdf

# 2. Complete journal, DOI, author information in YAML

# 3. Validate before processing
wqm ingest yaml ~/Papers/metadata_completion.yaml --dry-run

# 4. Process validated metadata
wqm ingest yaml ~/Papers/metadata_completion.yaml
```

### Example 3: Mixed Document Library
```bash
# 1. Generate comprehensive YAML
wqm ingest generate-yaml ~/Documents/Library --collection _mixed_library \\
  --format pdf,md,txt,epub

# 2. Complete metadata for different document types
#    Books: author, edition, ISBN
#    Articles: journal, DOI, publication date
#    Reports: organization, report number
#    Manuals: version, product information

# 3. Process in batches (complete some metadata, process, repeat)
wqm ingest yaml ~/Documents/Library/metadata_completion.yaml
```

## Advanced Features

### Document Type Detection
The system uses content analysis and filename patterns to detect document types:

- **Books**: Looks for chapters, ISBN, copyright information
- **Scientific Articles**: Detects abstracts, DOIs, journal references
- **Reports**: Identifies executive summaries, findings, recommendations
- **Manuals**: Recognizes step-by-step instructions, user guides

### Metadata Extraction
Enhanced extraction includes:

- **Title extraction**: From document headers and metadata
- **Author detection**: From "by" patterns and author fields
- **Publication info**: From copyright notices and publication data
- **ISBN/DOI extraction**: Using pattern matching
- **Date recognition**: From various date formats in content

### Iterative Processing
The workflow supports incremental completion:

1. Process documents with complete metadata
2. Update YAML with remaining incomplete documents
3. User completes additional metadata
4. Repeat until all documents are processed

### Error Handling
- Documents with extraction errors are flagged in the YAML
- Processing errors are reported with suggestions
- Validation ensures required fields are complete
- Graceful handling of unsupported file formats

## Best Practices

### Metadata Quality
- Provide accurate and consistent metadata
- Use standardized formats for dates and names
- Include relevant tags for better searchability
- Double-check detected metadata for accuracy

### File Organization
- Use descriptive filenames before processing
- Organize files in logical folder structures
- Keep related documents together
- Use consistent naming conventions

### Workflow Efficiency
- Process documents in logical batches
- Use dry-run to validate metadata before processing
- Keep YAML files for future reference
- Document your metadata conventions

### Collection Management
- Use meaningful collection names with '_' prefix
- Separate different types of libraries
- Consider version management for updated documents
- Regular backup of processed collections

## Troubleshooting

### Common Issues

**"Collection name must start with '_'"**
- Library collections must be prefixed with '_'
- Example: `_personal_library` not `personal_library`

**"No documents found"**
- Check file formats are supported (pdf, txt, md, epub)
- Verify files exist in specified directory
- Use `--format` to specify formats explicitly

**"Document type unknown with low confidence"**
- Add more descriptive content to documents
- Manually specify document type in YAML
- Review filename patterns for better detection

**"Missing required metadata"**
- Complete all fields marked with '?'
- Check schema requirements for document type
- Use empty string if value is genuinely unavailable

### Debug Information
- Use `--dry-run` to validate without processing
- Check extraction errors in YAML file
- Review confidence scores for type detection
- Examine detected metadata for accuracy

## Integration with Library Watching

The YAML metadata workflow integrates with the library folder watching system (Task 14):

1. Watched folders can generate YAML files automatically
2. Metadata completion triggers can be configured
3. Processed documents are tracked in collection metadata
4. Version management handles document updates

This creates a seamless pipeline from document addition to searchable library ingestion.