-- Language Support Database Schema
--
-- This schema defines four tables for tracking language support, LSP servers,
-- Tree-sitter parsers, and file metadata status in the workspace-qdrant-mcp system.
--
-- Relationships:
--   - languages: Central table defining supported languages and their tool status
--   - files_missing_metadata: Tracks files missing LSP/Tree-sitter metadata, references languages
--   - tools: Tracks LSP servers and Tree-sitter CLI installations
--   - language_support_version: Tracks YAML configuration hash for change detection

-- =============================================================================
-- Languages Table
-- =============================================================================
-- Central registry of programming languages with their associated tooling.
-- Tracks file extensions, LSP server details, and Tree-sitter grammar availability.
CREATE TABLE languages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    language_name TEXT UNIQUE NOT NULL,
    file_extensions TEXT,  -- JSON array of file extensions (e.g., [".py", ".pyi"])
    lsp_name TEXT,  -- LSP server identifier (e.g., "pylsp", "rust-analyzer")
    lsp_executable TEXT,  -- Executable name for LSP server (e.g., "pylsp")
    lsp_absolute_path TEXT,  -- Full path to LSP server executable
    lsp_missing BOOLEAN DEFAULT 1,  -- 1 if LSP server not found, 0 if available
    ts_grammar TEXT,  -- Tree-sitter grammar name (e.g., "python", "rust")
    ts_cli_absolute_path TEXT,  -- Full path to tree-sitter CLI executable
    ts_missing BOOLEAN DEFAULT 1,  -- 1 if Tree-sitter CLI not found, 0 if available
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for fast language name lookups
CREATE INDEX idx_languages_name ON languages(language_name);

-- Index for finding languages with missing LSP servers
CREATE INDEX idx_languages_lsp_missing ON languages(lsp_missing);

-- Index for finding languages with missing Tree-sitter support
CREATE INDEX idx_languages_ts_missing ON languages(ts_missing);

-- =============================================================================
-- Files Missing Metadata Table
-- =============================================================================
-- Tracks files that are missing LSP or Tree-sitter metadata extraction.
-- Links to languages table via foreign key to maintain referential integrity.
CREATE TABLE files_missing_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_absolute_path TEXT UNIQUE NOT NULL,
    language_name TEXT,  -- References languages.language_name
    branch TEXT,  -- Git branch context for the file
    missing_lsp_metadata BOOLEAN DEFAULT 0,  -- 1 if LSP metadata not extracted
    missing_ts_metadata BOOLEAN DEFAULT 0,  -- 1 if Tree-sitter metadata not extracted
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Foreign key constraint with CASCADE delete to maintain data integrity
    FOREIGN KEY (language_name) REFERENCES languages(language_name) ON DELETE CASCADE
);

-- Composite index for efficient queries filtering by language and missing metadata status
CREATE INDEX idx_files_missing_metadata_composite
ON files_missing_metadata(language_name, missing_lsp_metadata, missing_ts_metadata);

-- =============================================================================
-- Tools Table
-- =============================================================================
-- Tracks LSP servers and Tree-sitter CLI installations on the system.
-- Records tool paths, versions, and availability status.
CREATE TABLE tools (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_name TEXT UNIQUE NOT NULL,  -- Tool identifier (e.g., "pylsp", "tree-sitter-cli")
    tool_type TEXT CHECK(tool_type IN ('lsp_server', 'tree_sitter_cli')),  -- Tool category
    absolute_path TEXT,  -- Full path to tool executable
    version TEXT,  -- Tool version string
    missing BOOLEAN DEFAULT 1,  -- 1 if tool not found/unavailable, 0 if available
    last_check_at TIMESTAMP,  -- Last time tool availability was verified
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Composite index for finding missing tools by type
CREATE INDEX idx_tools_type_missing ON tools(tool_type, missing);

-- =============================================================================
-- Language Support Version Table
-- =============================================================================
-- Tracks the hash of language_support.yaml configuration file to detect changes.
-- Used to trigger re-initialization when language support configuration is updated.
CREATE TABLE language_support_version (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    yaml_hash TEXT UNIQUE NOT NULL,  -- SHA-256 hash of language_support.yaml
    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- When this configuration was loaded
    language_count INTEGER,  -- Number of languages in this configuration
    last_checked_at TIMESTAMP  -- Last time configuration was checked for changes
);
