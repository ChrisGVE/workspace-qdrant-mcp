from loguru import logger

"""
Source code file parser for extracting text content and metadata from programming files.

This module provides functionality to parse source code files across multiple programming
languages, extracting code content while preserving syntax highlighting information,
function/class definitions, and language-specific metadata.
"""

import re
from pathlib import Path
from typing import Any, Optional

try:
    from pygments import highlight
    from pygments.formatters import NullFormatter
    from pygments.lexers import get_lexer_by_name, get_lexer_for_filename
    from pygments.util import ClassNotFound

    PYGMENTS_AVAILABLE = True
except ImportError:
    PYGMENTS_AVAILABLE = False

try:
    from common.core.lsp_metadata_extractor import FileMetadata, LspMetadataExtractor
    LSP_AVAILABLE = True
except ImportError:
    LSP_AVAILABLE = False
    FileMetadata = None
    LspMetadataExtractor = None

from .base import DocumentParser, ParsedDocument
from .progress import ProgressTracker

# logger imported from loguru


class CodeParser(DocumentParser):
    """
    Enhanced parser for source code files with LSP integration.

    Extracts code content with syntax analysis, function/class detection,
    language-specific metadata extraction, and optional LSP-based code intelligence
    including symbols, relationships, and type information.
    """

    def __init__(self, lsp_extractor: Optional['LspMetadataExtractor'] = None):
        """
        Initialize code parser with optional LSP integration.

        Args:
            lsp_extractor: Optional LSP metadata extractor for enhanced code analysis
        """
        self.lsp_extractor = lsp_extractor
        self._check_lsp_availability()

    # Comprehensive mapping of file extensions to languages
    LANGUAGE_EXTENSIONS = {
        # Python
        ".py": "python",
        ".pyx": "python",
        ".pyi": "python",
        ".pyw": "python",
        # JavaScript/TypeScript
        ".js": "javascript",
        ".mjs": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        # Java/Kotlin/Scala
        ".java": "java",
        ".kt": "kotlin",
        ".kts": "kotlin",
        ".scala": "scala",
        # C/C++
        ".c": "c",
        ".h": "c",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".hpp": "cpp",
        ".hxx": "cpp",
        ".c++": "cpp",
        ".h++": "cpp",
        # C#
        ".cs": "csharp",
        # Go
        ".go": "go",
        # Rust
        ".rs": "rust",
        # Ruby
        ".rb": "ruby",
        ".rbw": "ruby",
        # PHP
        ".php": "php",
        ".phtml": "php",
        ".php3": "php",
        ".php4": "php",
        ".php5": "php",
        # Swift
        ".swift": "swift",
        # Objective-C
        ".m": "objc",
        ".mm": "objc",
        # R
        ".r": "r",
        ".R": "r",
        # MATLAB
        ".m": "matlab",  # Note: conflicts with Objective-C, will use context
        # Shell scripts
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "zsh",
        ".fish": "fish",
        ".ps1": "powershell",
        ".psm1": "powershell",
        # Web languages
        ".html": "html",
        ".htm": "html",
        ".xhtml": "html",
        ".css": "css",
        ".scss": "scss",
        ".sass": "sass",
        ".less": "less",
        ".xml": "xml",
        ".xsl": "xml",
        ".xsd": "xml",
        # SQL
        ".sql": "sql",
        # Config files
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".ini": "ini",
        ".cfg": "ini",
        ".conf": "ini",
        # Other languages
        ".lua": "lua",
        ".pl": "perl",
        ".pm": "perl",
        ".vim": "vim",
        ".dockerfile": "dockerfile",
        ".tf": "terraform",
        ".hcl": "hcl",
    }

    @property
    def supported_extensions(self) -> list[str]:
        """All supported code file extensions."""
        return list(self.LANGUAGE_EXTENSIONS.keys())

    @property
    def format_name(self) -> str:
        """Human-readable format name."""
        return "Source Code"

    def _check_availability(self) -> None:
        """Check if Pygments is available for syntax highlighting."""
        if not PYGMENTS_AVAILABLE:
            logger.warning("Pygments not available - basic code parsing only")

    def _check_lsp_availability(self) -> None:
        """Check if LSP integration is available."""
        if not LSP_AVAILABLE:
            logger.debug("LSP metadata extractor not available - basic code parsing only")
        elif self.lsp_extractor is None:
            logger.debug("No LSP extractor provided - enhanced analysis disabled")

    async def parse(self, file_path: str | Path, progress_tracker: ProgressTracker | None = None, **options: Any) -> ParsedDocument:
        """
        Parse source code file and extract content with enhanced LSP metadata.

        Args:
            file_path: Path to source code file
            progress_tracker: Optional progress tracker for monitoring
            **options: Parsing options
                - detect_functions: bool = True - Detect function/class definitions
                - include_comments: bool = True - Include comment extraction
                - syntax_highlighting: bool = False - Add syntax highlighting markers
                - max_line_length: int = 10000 - Skip files with extremely long lines
                - encoding: str = 'utf-8' - File encoding
                - enable_lsp: bool = True - Enable LSP-based analysis if available
                - lsp_timeout: float = 30.0 - LSP operation timeout

        Returns:
            ParsedDocument with extracted code and metadata

        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
            RuntimeError: If parsing fails
        """
        self._check_availability()
        self.validate_file(file_path)

        file_path = Path(file_path)

        try:
            # Parse options
            detect_functions = options.get("detect_functions", True)
            include_comments = options.get("include_comments", True)
            syntax_highlighting = options.get("syntax_highlighting", False)
            max_line_length = options.get("max_line_length", 10000)
            encoding = options.get("encoding", "utf-8")
            enable_lsp = options.get("enable_lsp", True)
            lsp_timeout = options.get("lsp_timeout", 30.0)

            # Update progress
            if progress_tracker:
                progress_tracker.update_phase("initializing", "Setting up code parsing")

            # Detect language
            if progress_tracker:
                progress_tracker.update_phase("analyzing", "Detecting programming language")
            language = await self._detect_language(file_path)

            # Read file content
            if progress_tracker:
                progress_tracker.update_phase("reading", "Reading source file")
            try:
                with open(file_path, encoding=encoding) as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Try with different encodings
                for fallback_encoding in ["latin1", "cp1252", "iso-8859-1"]:
                    try:
                        with open(file_path, encoding=fallback_encoding) as f:
                            content = f.read()
                        encoding = fallback_encoding
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise RuntimeError(
                        "Could not decode file with any common encoding"
                    )

            # Check for extremely long lines that might indicate binary content
            lines = content.split("\n")
            if any(len(line) > max_line_length for line in lines):
                raise RuntimeError(
                    f"File contains lines longer than {max_line_length} characters - likely binary"
                )

            # Extract basic metadata
            if progress_tracker:
                progress_tracker.update_phase("extracting", "Extracting basic metadata")
            metadata = await self._extract_metadata(content, language, file_path)

            # Enhanced content processing
            if progress_tracker:
                progress_tracker.update_phase("processing", "Processing code content")
            processed_content = content
            if syntax_highlighting and PYGMENTS_AVAILABLE:
                processed_content = await self._add_syntax_info(content, language)

            # Extract code structure if requested
            if detect_functions:
                code_structure = await self._extract_code_structure(content, language)
                metadata.update(code_structure)

            # Extract comments if requested
            if include_comments:
                comment_info = await self._extract_comments(content, language)
                metadata.update(comment_info)

            # LSP-enhanced analysis if available and enabled
            lsp_metadata = None
            if enable_lsp and self.lsp_extractor and LSP_AVAILABLE:
                if progress_tracker:
                    progress_tracker.update_phase("lsp_analysis", "Performing LSP-based code analysis")
                try:
                    lsp_metadata = await self._extract_lsp_metadata(file_path, lsp_timeout)
                    if lsp_metadata:
                        metadata.update(await self._integrate_lsp_metadata(lsp_metadata, metadata))
                        logger.debug(
                            f"LSP analysis completed for {file_path.name}: "
                            f"{len(lsp_metadata.symbols)} symbols, "
                            f"{len(lsp_metadata.relationships)} relationships"
                        )
                except Exception as e:
                    logger.warning(f"LSP analysis failed for {file_path}: {e}")
                    metadata["lsp_error"] = str(e)

            # Parsing information
            parsing_info = {
                "detected_language": language,
                "line_count": len(lines),
                "content_length": len(content),
                "encoding_used": encoding,
                "detect_functions": detect_functions,
                "include_comments": include_comments,
                "syntax_highlighting": syntax_highlighting,
                "lsp_enabled": enable_lsp and self.lsp_extractor is not None,
                "lsp_analysis_success": lsp_metadata is not None,
            }

            if progress_tracker:
                progress_tracker.update_phase("finalizing", "Creating parsed document")

            logger.info(
                f"Successfully parsed {language} code: {file_path.name} "
                f"({parsing_info['line_count']} lines, "
                f"{parsing_info['content_length']:,} characters)"
            )

            return ParsedDocument.create(
                content=processed_content,
                file_path=file_path,
                file_type="code",
                additional_metadata=metadata,
                parsing_info=parsing_info,
            )

        except Exception as e:
            logger.error(f"Failed to parse code file {file_path}: {e}")
            raise RuntimeError(f"Code parsing failed: {e}") from e

    async def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension and content."""
        extension = file_path.suffix.lower()

        # Primary detection by extension
        if extension in self.LANGUAGE_EXTENSIONS:
            return self.LANGUAGE_EXTENSIONS[extension]

        # Fallback: try to detect by filename patterns
        name = file_path.name.lower()

        # Special cases
        if name in ["dockerfile", "containerfile"]:
            return "dockerfile"
        elif name in ["makefile", "gnumakefile"]:
            return "makefile"
        elif name.startswith("."):
            if name in [".gitignore", ".dockerignore"]:
                return "text"
            elif name in [".bashrc", ".zshrc", ".profile"]:
                return "bash"

        # If Pygments is available, try lexer detection
        if PYGMENTS_AVAILABLE:
            try:
                lexer = get_lexer_for_filename(str(file_path))
                return lexer.name.lower()
            except ClassNotFound:
                pass

        # Default fallback
        return "text"

    async def _extract_metadata(
        self, content: str, language: str, file_path: Path
    ) -> dict[str, str | int | float | bool]:
        """Extract metadata from code content."""
        lines = content.split("\n")
        metadata = {
            "programming_language": language,
            "line_count": len(lines),
            "non_empty_lines": sum(1 for line in lines if line.strip()),
            "character_count": len(content),
        }

        # Count different types of lines
        empty_lines = sum(1 for line in lines if not line.strip())
        comment_lines = await self._count_comment_lines(lines, language)

        metadata.update(
            {
                "empty_lines": empty_lines,
                "comment_lines": comment_lines,
                "code_lines": len(lines) - empty_lines - comment_lines,
                "comment_ratio": comment_lines / max(1, len(lines)),
            }
        )

        # Language-specific analysis
        if language in ["python", "javascript", "typescript", "java", "csharp", "cpp"]:
            metadata.update(await self._extract_structure_metadata(content, language))

        return metadata

    async def _count_comment_lines(self, lines: list[str], language: str) -> int:
        """Count comment lines based on language syntax."""
        comment_count = 0

        # Define comment patterns by language
        single_line_comments = {
            "python": ["#"],
            "javascript": ["//"],
            "typescript": ["//"],
            "java": ["//"],
            "csharp": ["//"],
            "cpp": ["//"],
            "c": ["//"],
            "go": ["//"],
            "rust": ["//"],
            "swift": ["//"],
            "bash": ["#"],
            "shell": ["#"],
            "ruby": ["#"],
            "r": ["#"],
            "sql": ["--"],
            "lua": ["--"],
        }

        patterns = single_line_comments.get(language, [])

        for line in lines:
            stripped = line.strip()
            if any(stripped.startswith(pattern) for pattern in patterns):
                comment_count += 1

        return comment_count

    async def _extract_structure_metadata(
        self, content: str, language: str
    ) -> dict[str, int]:
        """Extract structural metadata (function count, class count, etc.)."""
        metadata = {}

        if language == "python":
            metadata.update(await self._extract_python_structure(content))
        elif language in ["javascript", "typescript"]:
            metadata.update(await self._extract_js_structure(content))
        elif language == "java":
            metadata.update(await self._extract_java_structure(content))

        return metadata

    async def _extract_python_structure(self, content: str) -> dict[str, int]:
        """Extract Python-specific structure information."""
        # Function definitions
        func_pattern = r"^[ \t]*def\s+\w+\s*\("
        function_count = len(re.findall(func_pattern, content, re.MULTILINE))

        # Class definitions
        class_pattern = r"^[ \t]*class\s+\w+\s*[\(:]"
        class_count = len(re.findall(class_pattern, content, re.MULTILINE))

        # Import statements
        import_pattern = r"^[ \t]*(?:import|from)\s+"
        import_count = len(re.findall(import_pattern, content, re.MULTILINE))

        return {
            "function_count": function_count,
            "class_count": class_count,
            "import_count": import_count,
        }

    async def _extract_js_structure(self, content: str) -> dict[str, int]:
        """Extract JavaScript/TypeScript structure information."""
        # Function declarations and expressions
        func_patterns = [
            r"\bfunction\s+\w+\s*\(",  # function declarations
            r"\w+\s*=\s*function\s*\(",  # function expressions
            r"\w+\s*:\s*function\s*\(",  # object method functions
            r"\w+\s*=\s*\([^)]*\)\s*=>",  # arrow functions
        ]

        function_count = 0
        for pattern in func_patterns:
            function_count += len(re.findall(pattern, content))

        # Class definitions (ES6+)
        class_pattern = r"\bclass\s+\w+"
        class_count = len(re.findall(class_pattern, content))

        # Import statements
        import_patterns = [
            r"\bimport\s+",
            r"\brequire\s*\(",
        ]

        import_count = 0
        for pattern in import_patterns:
            import_count += len(re.findall(pattern, content))

        return {
            "function_count": function_count,
            "class_count": class_count,
            "import_count": import_count,
        }

    async def _extract_java_structure(self, content: str) -> dict[str, int]:
        """Extract Java structure information."""
        # Method definitions
        method_pattern = (
            r"(?:public|private|protected|static|\s)+[\w<>\[\]]+\s+\w+\s*\([^)]*\)\s*\{"
        )
        method_count = len(re.findall(method_pattern, content))

        # Class definitions
        class_pattern = r"\b(?:public|private|protected)?\s*class\s+\w+"
        class_count = len(re.findall(class_pattern, content))

        # Interface definitions
        interface_pattern = r"\b(?:public|private|protected)?\s*interface\s+\w+"
        interface_count = len(re.findall(interface_pattern, content))

        # Import statements
        import_pattern = r"\bimport\s+"
        import_count = len(re.findall(import_pattern, content))

        return {
            "method_count": method_count,
            "class_count": class_count,
            "interface_count": interface_count,
            "import_count": import_count,
        }

    async def _extract_code_structure(
        self, content: str, language: str
    ) -> dict[str, Any]:
        """Extract detailed code structure information."""
        structure_info = await self._extract_structure_metadata(content, language)

        # Add complexity metrics
        lines = content.split("\n")
        indentation_levels = []

        for line in lines:
            if line.strip():  # Skip empty lines
                indent = len(line) - len(line.lstrip())
                indentation_levels.append(indent)

        if indentation_levels:
            structure_info.update(
                {
                    "max_indentation": max(indentation_levels),
                    "avg_indentation": sum(indentation_levels)
                    / len(indentation_levels),
                }
            )

        return structure_info

    async def _extract_comments(self, content: str, language: str) -> dict[str, Any]:
        """Extract comment analysis information."""
        lines = content.split("\n")
        comment_lines = await self._count_comment_lines(lines, language)

        return {
            "has_comments": comment_lines > 0,
            "comment_density": comment_lines / max(1, len(lines)),
        }

    async def _add_syntax_info(self, content: str, language: str) -> str:
        """Add syntax highlighting information using Pygments."""
        try:
            lexer = get_lexer_by_name(language)
            # Use NullFormatter to get tokens without formatting
            list(lexer.get_tokens(content))

            # For now, just return original content
            # Could add token type information as comments in the future
            return content

        except ClassNotFound:
            return content

    async def _extract_lsp_metadata(self, file_path: Path, timeout: float = 30.0) -> Optional['FileMetadata']:
        """
        Extract LSP-based metadata from the source file.

        Args:
            file_path: Path to the source file
            timeout: LSP operation timeout

        Returns:
            FileMetadata object with LSP analysis results or None if failed
        """
        if not self.lsp_extractor or not LSP_AVAILABLE:
            return None

        try:
            # Ensure LSP extractor is initialized
            if not self.lsp_extractor._initialized:
                await self.lsp_extractor.initialize(file_path.parent)

            # Extract metadata with timeout
            return await self.lsp_extractor.extract_file_metadata(file_path)

        except Exception as e:
            logger.debug(f"LSP metadata extraction failed: {e}")
            return None

    async def _integrate_lsp_metadata(self, lsp_metadata: 'FileMetadata', basic_metadata: dict) -> dict:
        """
        Integrate LSP metadata with basic parsing metadata.

        Args:
            lsp_metadata: LSP-extracted metadata
            basic_metadata: Basic parsing metadata

        Returns:
            Enhanced metadata dictionary
        """
        enhanced_metadata = basic_metadata.copy()

        # Add LSP symbol information
        if lsp_metadata.symbols:
            symbol_info = {
                "lsp_symbols_count": len(lsp_metadata.symbols),
                "lsp_functions": [],
                "lsp_classes": [],
                "lsp_variables": [],
                "lsp_imports": lsp_metadata.imports,
                "lsp_exports": lsp_metadata.exports,
            }

            # Categorize symbols by type
            for symbol in lsp_metadata.symbols:
                symbol_data = {
                    "name": symbol.name,
                    "kind": symbol.kind.name,
                    "line": symbol.range.start.line + 1,  # Convert to 1-based
                    "signature": symbol.get_signature(),
                }

                # Add documentation if available
                if symbol.documentation and symbol.documentation.docstring:
                    symbol_data["documentation"] = symbol.documentation.docstring

                # Add type information if available
                if symbol.type_info:
                    if symbol.type_info.type_name:
                        symbol_data["type"] = symbol.type_info.type_name
                    if symbol.type_info.return_type:
                        symbol_data["return_type"] = symbol.type_info.return_type

                # Categorize by symbol kind
                if symbol.kind.name in ["FUNCTION", "METHOD", "CONSTRUCTOR"]:
                    symbol_info["lsp_functions"].append(symbol_data)
                elif symbol.kind.name in ["CLASS", "INTERFACE", "STRUCT"]:
                    symbol_info["lsp_classes"].append(symbol_data)
                elif symbol.kind.name in ["VARIABLE", "CONSTANT", "FIELD", "PROPERTY"]:
                    symbol_info["lsp_variables"].append(symbol_data)

            enhanced_metadata.update(symbol_info)

        # Add relationship information
        if lsp_metadata.relationships:
            enhanced_metadata["lsp_relationships_count"] = len(lsp_metadata.relationships)
            enhanced_metadata["lsp_relationships"] = [
                {
                    "from": rel.from_symbol,
                    "to": rel.to_symbol,
                    "type": rel.relationship_type.value,
                }
                for rel in lsp_metadata.relationships
            ]

        # Add file-level documentation
        if lsp_metadata.file_docstring:
            enhanced_metadata["file_docstring"] = lsp_metadata.file_docstring

        if lsp_metadata.file_comments:
            enhanced_metadata["file_comments"] = lsp_metadata.file_comments

        # Add extraction statistics
        if lsp_metadata.extraction_errors:
            enhanced_metadata["lsp_extraction_errors"] = lsp_metadata.extraction_errors

        enhanced_metadata["lsp_server"] = lsp_metadata.lsp_server
        enhanced_metadata["lsp_extraction_timestamp"] = lsp_metadata.extraction_timestamp

        return enhanced_metadata

    def get_parsing_options(self) -> dict[str, dict[str, Any]]:
        """Get available parsing options for code files."""
        return {
            "detect_functions": {
                "type": bool,
                "default": True,
                "description": "Detect and count function/class definitions",
            },
            "include_comments": {
                "type": bool,
                "default": True,
                "description": "Include comment analysis in metadata",
            },
            "syntax_highlighting": {
                "type": bool,
                "default": False,
                "description": "Add syntax highlighting information (requires Pygments)",
            },
            "max_line_length": {
                "type": int,
                "default": 10000,
                "description": "Maximum line length before treating as binary file",
            },
            "encoding": {
                "type": str,
                "default": "utf-8",
                "description": "File encoding (utf-8, latin1, etc.)",
            },
            "enable_lsp": {
                "type": bool,
                "default": True,
                "description": "Enable LSP-based code analysis for enhanced metadata",
            },
            "lsp_timeout": {
                "type": float,
                "default": 30.0,
                "description": "Timeout for LSP operations in seconds",
            },
        }
