"""
Source code file parser for extracting text content and metadata from programming files.

This module provides functionality to parse source code files across multiple programming
languages, extracting code content while preserving syntax highlighting information,
function/class definitions, and language-specific metadata.
"""

import logging
from pathlib import Path
from typing import Any
import re

try:
    from pygments import highlight
    from pygments.lexers import get_lexer_for_filename, get_lexer_by_name
    from pygments.formatters import NullFormatter
    from pygments.util import ClassNotFound
    PYGMENTS_AVAILABLE = True
except ImportError:
    PYGMENTS_AVAILABLE = False

from .base import DocumentParser, ParsedDocument

logger = logging.getLogger(__name__)


class CodeParser(DocumentParser):
    """
    Parser for source code files across multiple programming languages.
    
    Extracts code content with syntax analysis, function/class detection,
    and language-specific metadata extraction.
    """

    # Comprehensive mapping of file extensions to languages
    LANGUAGE_EXTENSIONS = {
        # Python
        '.py': 'python', '.pyx': 'python', '.pyi': 'python', '.pyw': 'python',
        # JavaScript/TypeScript
        '.js': 'javascript', '.mjs': 'javascript', '.jsx': 'javascript',
        '.ts': 'typescript', '.tsx': 'typescript',
        # Java/Kotlin/Scala
        '.java': 'java', '.kt': 'kotlin', '.kts': 'kotlin', '.scala': 'scala',
        # C/C++
        '.c': 'c', '.h': 'c', '.cpp': 'cpp', '.cc': 'cpp', '.cxx': 'cpp',
        '.hpp': 'cpp', '.hxx': 'cpp', '.c++': 'cpp', '.h++': 'cpp',
        # C#
        '.cs': 'csharp',
        # Go
        '.go': 'go',
        # Rust
        '.rs': 'rust',
        # Ruby
        '.rb': 'ruby', '.rbw': 'ruby',
        # PHP
        '.php': 'php', '.phtml': 'php', '.php3': 'php', '.php4': 'php', '.php5': 'php',
        # Swift
        '.swift': 'swift',
        # Objective-C
        '.m': 'objc', '.mm': 'objc',
        # R
        '.r': 'r', '.R': 'r',
        # MATLAB
        '.m': 'matlab',  # Note: conflicts with Objective-C, will use context
        # Shell scripts
        '.sh': 'bash', '.bash': 'bash', '.zsh': 'zsh', '.fish': 'fish',
        '.ps1': 'powershell', '.psm1': 'powershell',
        # Web languages
        '.html': 'html', '.htm': 'html', '.xhtml': 'html',
        '.css': 'css', '.scss': 'scss', '.sass': 'sass', '.less': 'less',
        '.xml': 'xml', '.xsl': 'xml', '.xsd': 'xml',
        # SQL
        '.sql': 'sql',
        # Config files
        '.json': 'json', '.yaml': 'yaml', '.yml': 'yaml', '.toml': 'toml',
        '.ini': 'ini', '.cfg': 'ini', '.conf': 'ini',
        # Other languages
        '.lua': 'lua', '.pl': 'perl', '.pm': 'perl',
        '.vim': 'vim', '.dockerfile': 'dockerfile',
        '.tf': 'terraform', '.hcl': 'hcl',
    }

    @property
    def supported_extensions(self) -> list[str]:
        """All supported code file extensions."""
        return list(self.LANGUAGE_EXTENSIONS.keys())

    @property
    def format_name(self) -> str:
        """Human-readable format name."""
        return 'Source Code'

    def _check_availability(self) -> None:
        """Check if Pygments is available for syntax highlighting."""
        if not PYGMENTS_AVAILABLE:
            logger.warning("Pygments not available - basic code parsing only")

    async def parse(self, file_path: str | Path, **options: Any) -> ParsedDocument:
        """
        Parse source code file and extract content with metadata.

        Args:
            file_path: Path to source code file
            **options: Parsing options
                - detect_functions: bool = True - Detect function/class definitions
                - include_comments: bool = True - Include comment extraction
                - syntax_highlighting: bool = False - Add syntax highlighting markers
                - max_line_length: int = 10000 - Skip files with extremely long lines
                - encoding: str = 'utf-8' - File encoding

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
            detect_functions = options.get('detect_functions', True)
            include_comments = options.get('include_comments', True)
            syntax_highlighting = options.get('syntax_highlighting', False)
            max_line_length = options.get('max_line_length', 10000)
            encoding = options.get('encoding', 'utf-8')
            
            # Detect language
            language = await self._detect_language(file_path)
            
            # Read file content
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Try with different encodings
                for fallback_encoding in ['latin1', 'cp1252', 'iso-8859-1']:
                    try:
                        with open(file_path, 'r', encoding=fallback_encoding) as f:
                            content = f.read()
                        encoding = fallback_encoding
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise RuntimeError(f"Could not decode file with any common encoding")
            
            # Check for extremely long lines that might indicate binary content
            lines = content.split('\n')
            if any(len(line) > max_line_length for line in lines):
                raise RuntimeError(f"File contains lines longer than {max_line_length} characters - likely binary")
            
            # Extract metadata
            metadata = await self._extract_metadata(content, language, file_path)
            
            # Enhanced content processing
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
            
            # Parsing information
            parsing_info = {
                "detected_language": language,
                "line_count": len(lines),
                "content_length": len(content),
                "encoding_used": encoding,
                "detect_functions": detect_functions,
                "include_comments": include_comments,
                "syntax_highlighting": syntax_highlighting
            }
            
            logger.info(f"Successfully parsed {language} code: {file_path.name} "
                       f"({parsing_info['line_count']} lines, "
                       f"{parsing_info['content_length']:,} characters)")
            
            return ParsedDocument.create(
                content=processed_content,
                file_path=file_path,
                file_type='code',
                additional_metadata=metadata,
                parsing_info=parsing_info
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
        if name in ['dockerfile', 'containerfile']:
            return 'dockerfile'
        elif name in ['makefile', 'gnumakefile']:
            return 'makefile'
        elif name.startswith('.'):
            if name in ['.gitignore', '.dockerignore']:
                return 'text'
            elif name in ['.bashrc', '.zshrc', '.profile']:
                return 'bash'
        
        # If Pygments is available, try lexer detection
        if PYGMENTS_AVAILABLE:
            try:
                lexer = get_lexer_for_filename(str(file_path))
                return lexer.name.lower()
            except ClassNotFound:
                pass
        
        # Default fallback
        return 'text'

    async def _extract_metadata(self, content: str, language: str, file_path: Path) -> dict[str, str | int | float | bool]:
        """Extract metadata from code content."""
        lines = content.split('\n')
        metadata = {
            'programming_language': language,
            'line_count': len(lines),
            'non_empty_lines': sum(1 for line in lines if line.strip()),
            'character_count': len(content),
        }
        
        # Count different types of lines
        empty_lines = sum(1 for line in lines if not line.strip())
        comment_lines = await self._count_comment_lines(lines, language)
        
        metadata.update({
            'empty_lines': empty_lines,
            'comment_lines': comment_lines,
            'code_lines': len(lines) - empty_lines - comment_lines,
            'comment_ratio': comment_lines / max(1, len(lines)),
        })
        
        # Language-specific analysis
        if language in ['python', 'javascript', 'typescript', 'java', 'csharp', 'cpp']:
            metadata.update(await self._extract_structure_metadata(content, language))
        
        return metadata

    async def _count_comment_lines(self, lines: list[str], language: str) -> int:
        """Count comment lines based on language syntax."""
        comment_count = 0
        
        # Define comment patterns by language
        single_line_comments = {
            'python': ['#'],
            'javascript': ['//'], 'typescript': ['//'],
            'java': ['//'], 'csharp': ['//'], 'cpp': ['//'], 'c': ['//'],
            'go': ['//'], 'rust': ['//'], 'swift': ['//'],
            'bash': ['#'], 'shell': ['#'],
            'ruby': ['#'], 'r': ['#'],
            'sql': ['--'],
            'lua': ['--'],
        }
        
        patterns = single_line_comments.get(language, [])
        
        for line in lines:
            stripped = line.strip()
            if any(stripped.startswith(pattern) for pattern in patterns):
                comment_count += 1
        
        return comment_count

    async def _extract_structure_metadata(self, content: str, language: str) -> dict[str, int]:
        """Extract structural metadata (function count, class count, etc.)."""
        metadata = {}
        
        if language == 'python':
            metadata.update(await self._extract_python_structure(content))
        elif language in ['javascript', 'typescript']:
            metadata.update(await self._extract_js_structure(content))
        elif language == 'java':
            metadata.update(await self._extract_java_structure(content))
        
        return metadata

    async def _extract_python_structure(self, content: str) -> dict[str, int]:
        """Extract Python-specific structure information."""
        # Function definitions
        func_pattern = r'^[ \t]*def\s+\w+\s*\('
        function_count = len(re.findall(func_pattern, content, re.MULTILINE))
        
        # Class definitions
        class_pattern = r'^[ \t]*class\s+\w+\s*[\(:]'
        class_count = len(re.findall(class_pattern, content, re.MULTILINE))
        
        # Import statements
        import_pattern = r'^[ \t]*(?:import|from)\s+'
        import_count = len(re.findall(import_pattern, content, re.MULTILINE))
        
        return {
            'function_count': function_count,
            'class_count': class_count,
            'import_count': import_count,
        }

    async def _extract_js_structure(self, content: str) -> dict[str, int]:
        """Extract JavaScript/TypeScript structure information."""
        # Function declarations and expressions
        func_patterns = [
            r'\bfunction\s+\w+\s*\(',  # function declarations
            r'\w+\s*=\s*function\s*\(',  # function expressions
            r'\w+\s*:\s*function\s*\(',  # object method functions
            r'\w+\s*=\s*\([^)]*\)\s*=>', # arrow functions
        ]
        
        function_count = 0
        for pattern in func_patterns:
            function_count += len(re.findall(pattern, content))
        
        # Class definitions (ES6+)
        class_pattern = r'\bclass\s+\w+'
        class_count = len(re.findall(class_pattern, content))
        
        # Import statements
        import_patterns = [
            r'\bimport\s+',
            r'\brequire\s*\(',
        ]
        
        import_count = 0
        for pattern in import_patterns:
            import_count += len(re.findall(pattern, content))
        
        return {
            'function_count': function_count,
            'class_count': class_count,
            'import_count': import_count,
        }

    async def _extract_java_structure(self, content: str) -> dict[str, int]:
        """Extract Java structure information."""
        # Method definitions
        method_pattern = r'(?:public|private|protected|static|\s)+[\w<>\[\]]+\s+\w+\s*\([^)]*\)\s*\{'
        method_count = len(re.findall(method_pattern, content))
        
        # Class definitions
        class_pattern = r'\b(?:public|private|protected)?\s*class\s+\w+'
        class_count = len(re.findall(class_pattern, content))
        
        # Interface definitions
        interface_pattern = r'\b(?:public|private|protected)?\s*interface\s+\w+'
        interface_count = len(re.findall(interface_pattern, content))
        
        # Import statements
        import_pattern = r'\bimport\s+'
        import_count = len(re.findall(import_pattern, content))
        
        return {
            'method_count': method_count,
            'class_count': class_count,
            'interface_count': interface_count,
            'import_count': import_count,
        }

    async def _extract_code_structure(self, content: str, language: str) -> dict[str, Any]:
        """Extract detailed code structure information."""
        structure_info = await self._extract_structure_metadata(content, language)
        
        # Add complexity metrics
        lines = content.split('\n')
        indentation_levels = []
        
        for line in lines:
            if line.strip():  # Skip empty lines
                indent = len(line) - len(line.lstrip())
                indentation_levels.append(indent)
        
        if indentation_levels:
            structure_info.update({
                'max_indentation': max(indentation_levels),
                'avg_indentation': sum(indentation_levels) / len(indentation_levels),
            })
        
        return structure_info

    async def _extract_comments(self, content: str, language: str) -> dict[str, Any]:
        """Extract comment analysis information."""
        lines = content.split('\n')
        comment_lines = await self._count_comment_lines(lines, language)
        
        return {
            'has_comments': comment_lines > 0,
            'comment_density': comment_lines / max(1, len(lines)),
        }

    async def _add_syntax_info(self, content: str, language: str) -> str:
        """Add syntax highlighting information using Pygments."""
        try:
            lexer = get_lexer_by_name(language)
            # Use NullFormatter to get tokens without formatting
            tokens = list(lexer.get_tokens(content))
            
            # For now, just return original content
            # Could add token type information as comments in the future
            return content
            
        except ClassNotFound:
            return content

    def get_parsing_options(self) -> dict[str, dict[str, Any]]:
        """Get available parsing options for code files."""
        return {
            'detect_functions': {
                'type': bool,
                'default': True,
                'description': 'Detect and count function/class definitions'
            },
            'include_comments': {
                'type': bool,
                'default': True,
                'description': 'Include comment analysis in metadata'
            },
            'syntax_highlighting': {
                'type': bool,
                'default': False,
                'description': 'Add syntax highlighting information (requires Pygments)'
            },
            'max_line_length': {
                'type': int,
                'default': 10000,
                'description': 'Maximum line length before treating as binary file'
            },
            'encoding': {
                'type': str,
                'default': 'utf-8',
                'description': 'File encoding (utf-8, latin1, etc.)'
            }
        }