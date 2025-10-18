"""File type classification for metadata-based routing.

This module provides file type classification to replace collection-based routing
with metadata-based differentiation. Part of Task 374 ingestion pipeline refactoring.

File Types:
- code: Source code files (.py, .rs, .js, etc.)
- test: Test files (test_*, *_test.*, spec.*)
- docs: Documentation (.md, .rst, .txt, .adoc)
- config: Configuration files (.yaml, .json, .toml, .ini)
- data: Data files (.csv, .parquet, .json, .xml)
- build: Build artifacts (.whl, .tar.gz, .zip, .jar)
- other: Unclassified files
"""

from pathlib import Path
from typing import Set


# File extension sets for classification
CODE_EXTENSIONS: Set[str] = {
    '.py', '.pyx', '.pyi',  # Python
    '.rs',  # Rust
    '.js', '.jsx', '.mjs', '.cjs',  # JavaScript
    '.ts', '.tsx', '.d.ts',  # TypeScript
    '.go',  # Go
    '.java', '.kt', '.scala',  # JVM languages
    '.c', '.cpp', '.cxx', '.cc', '.h', '.hpp', '.hxx',  # C/C++
    '.cs', '.fs', '.vb',  # .NET languages
    '.rb', '.erb',  # Ruby
    '.php', '.phtml',  # PHP
    '.swift',  # Swift
    '.m', '.mm',  # Objective-C
    '.sh', '.bash', '.zsh', '.fish',  # Shell scripts
    '.sql', '.ddl', '.dml',  # SQL
    '.r', '.R',  # R
    '.jl',  # Julia
    '.hs',  # Haskell
    '.erl', '.ex', '.exs',  # Erlang/Elixir
    '.clj', '.cljs',  # Clojure
    '.ml', '.mli',  # OCaml
    '.lua',  # Lua
    '.vim',  # Vimscript
    '.el',  # Emacs Lisp
}

DOCS_EXTENSIONS: Set[str] = {
    '.md', '.markdown',  # Markdown
    '.rst', '.rest',  # reStructuredText
    '.txt', '.text',  # Plain text
    '.adoc', '.asciidoc',  # AsciiDoc
    '.org',  # Org mode
    '.tex',  # LaTeX
    '.pdf',  # PDF
    '.epub',  # EPUB
    '.docx', '.doc',  # Word
    '.odt',  # OpenDocument Text
    '.rtf',  # Rich Text Format
}

CONFIG_EXTENSIONS: Set[str] = {
    '.yaml', '.yml',  # YAML
    '.json', '.jsonc', '.json5',  # JSON
    '.toml',  # TOML
    '.ini',  # INI
    '.conf', '.cfg', '.config',  # Generic config
    '.env',  # Environment files
    '.properties',  # Java properties
    '.xml',  # XML (can be config)
    '.plist',  # macOS property list
    '.editorconfig',  # EditorConfig
    '.gitconfig', '.gitignore', '.gitattributes',  # Git config
}

DATA_EXTENSIONS: Set[str] = {
    '.csv', '.tsv',  # CSV/TSV
    '.parquet',  # Parquet
    '.json', '.jsonl', '.ndjson',  # JSON data (overlaps with config, intent-based)
    '.xml',  # XML data (overlaps with config)
    '.arrow',  # Apache Arrow
    '.feather',  # Feather format
    '.hdf5', '.h5',  # HDF5
    '.db', '.sqlite', '.sqlite3',  # SQLite
    '.pkl', '.pickle',  # Python pickle
    '.npy', '.npz',  # NumPy arrays
    '.mat',  # MATLAB data
    '.rds', '.rdata',  # R data
}

BUILD_EXTENSIONS: Set[str] = {
    '.whl',  # Python wheel
    '.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tar.xz', '.txz',  # Tarballs
    '.zip',  # ZIP
    '.jar', '.war', '.ear',  # Java archives
    '.so', '.dylib', '.dll',  # Shared libraries
    '.a', '.lib',  # Static libraries
    '.o', '.obj',  # Object files
    '.exe', '.app',  # Executables
    '.deb', '.rpm',  # Package formats
    '.dmg',  # macOS disk image
    '.iso',  # ISO image
}


def _get_compound_extension(filename: str) -> str:
    """Extract compound extension from filename if present.

    Handles multi-part extensions like .tar.gz, .tar.bz2, etc.

    Args:
        filename: Lowercase filename

    Returns:
        Compound extension if found, empty string otherwise

    Examples:
        >>> _get_compound_extension("package.tar.gz")
        '.tar.gz'
        >>> _get_compound_extension("archive.tgz")
        '.tgz'
        >>> _get_compound_extension("file.txt")
        ''
    """
    # Common compound extensions to check
    compound_extensions = [
        '.tar.gz', '.tar.bz2', '.tar.xz',
        '.tgz', '.tbz2', '.txz',
    ]

    for ext in compound_extensions:
        if filename.endswith(ext):
            return ext

    return ''


def determine_file_type(file_path: Path) -> str:
    """Determine file type for metadata classification.

    Classification priority:
    1. Test files (checked first to avoid misclassification as code)
    2. Documentation files
    3. Configuration files
    4. Code files
    5. Data files
    6. Build artifacts
    7. Other (fallback)

    Args:
        file_path: Path to the file to classify

    Returns:
        File type string: "code", "test", "docs", "config", "data", "build", or "other"

    Examples:
        >>> determine_file_type(Path("test_auth.py"))
        'test'
        >>> determine_file_type(Path("README.md"))
        'docs'
        >>> determine_file_type(Path("main.py"))
        'code'
        >>> determine_file_type(Path("config.yaml"))
        'config'
    """
    name = file_path.name.lower()
    ext = file_path.suffix.lower()

    # Handle compound extensions (e.g., .tar.gz, .tar.bz2)
    compound_ext = _get_compound_extension(name)
    if compound_ext:
        ext = compound_ext

    # Handle dotfiles without extensions (e.g., .env, .gitconfig)
    # Also handle .env variants like .env.local, .env.development
    if not ext and name.startswith('.'):
        ext = name  # Use full name as extension for dotfiles
    elif name.startswith('.env'):
        # Special handling for .env.* files
        ext = '.env'

    # Priority 1: Test files (must check before code classification)
    if _is_test_file(name, ext):
        return "test"

    # Priority 2: Documentation
    if ext in DOCS_EXTENSIONS:
        return "docs"

    # Priority 3: Configuration
    # Special handling for JSON/XML which can be data or config
    if ext in CONFIG_EXTENSIONS:
        # If it's JSON/XML in typical config locations, classify as config
        if ext in {'.json', '.xml'}:
            # Check parent directory names for config indicators
            parent_parts = [p.lower() for p in file_path.parts]
            config_indicators = {'config', 'conf', 'settings', '.github', '.vscode'}
            if any(indicator in parent_parts for indicator in config_indicators):
                return "config"
            # Otherwise, treat as data
            return "data"
        return "config"

    # Priority 4: Code files
    if ext in CODE_EXTENSIONS:
        return "code"

    # Priority 5: Data files
    if ext in DATA_EXTENSIONS:
        return "data"

    # Priority 6: Build artifacts
    if ext in BUILD_EXTENSIONS:
        return "build"

    # Priority 7: Fallback to "other"
    return "other"


def _is_test_file(name: str, ext: str) -> bool:
    """Check if file is a test file based on naming conventions.

    Detects common test file patterns:
    - test_*.py, test_*.rs, etc.
    - *_test.py, *_test.rs, etc.
    - *.test.js, *.spec.ts, etc.
    - Tests in __tests__ or test directories are handled by path logic

    Args:
        name: Lowercase filename
        ext: Lowercase file extension

    Returns:
        True if file appears to be a test file, False otherwise
    """
    # Common test file prefixes/suffixes
    if name.startswith('test_'):
        return True

    # Common test file patterns
    name_without_ext = name.rsplit('.', 1)[0] if '.' in name else name

    if name_without_ext.endswith('_test'):
        return True

    if name_without_ext.endswith('.test') or name_without_ext.endswith('.spec'):
        return True

    # Spec files (common in JS/TS ecosystems)
    if '.spec.' in name or '.test.' in name:
        return True

    # Special test file names
    test_filenames = {'conftest', 'test', 'tests'}
    if name_without_ext in test_filenames and ext in CODE_EXTENSIONS:
        return True

    return False


def is_test_directory(directory_path: Path) -> bool:
    """Check if a directory is a test directory.

    Common test directory names:
    - tests, test, __tests__
    - spec, specs
    - integration, e2e, unit

    Args:
        directory_path: Path to directory

    Returns:
        True if directory name suggests it contains tests, False otherwise
    """
    dir_name = directory_path.name.lower()
    test_dir_names = {
        'tests', 'test', '__tests__',
        'spec', 'specs',
        'integration', 'e2e', 'unit',
        'functional', 'acceptance'
    }
    return dir_name in test_dir_names
