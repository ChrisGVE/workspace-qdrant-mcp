"""
Test data collection from actual workspace-qdrant-mcp codebase.

Collects real Python code, functions, classes, and documentation
to use as test corpus for realistic recall/precision measurements.
"""

import ast
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class CodeSymbol:
    """Represents a code symbol (function, class, method)."""

    name: str
    type: str  # 'function', 'class', 'method'
    file_path: str
    line_number: int
    docstring: Optional[str] = None
    signature: Optional[str] = None
    source_code: Optional[str] = None
    parent_class: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CodeChunk:
    """Represents a chunk of code or documentation."""

    content: str
    file_path: str
    chunk_type: str  # 'code', 'docstring', 'comment', 'documentation'
    line_start: int
    line_end: int
    symbols: list[str] = field(default_factory=list)  # Related symbols
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchGroundTruth:
    """Ground truth data for search quality evaluation."""

    query: str
    query_type: str  # 'symbol', 'semantic', 'exact', 'hybrid'
    expected_results: list[str]  # Expected document IDs
    expected_symbols: list[str] = field(default_factory=list)
    relevance_scores: dict[str, float] = field(default_factory=dict)


class TestDataCollector:
    """
    Collects real codebase data for comprehensive testing.

    Scans the workspace-qdrant-mcp source code to extract:
    - Python functions, classes, and methods
    - Docstrings and comments
    - Documentation files
    - Project structure information
    """

    def __init__(self, source_root: Path):
        self.source_root = Path(source_root)
        self.symbols: list[CodeSymbol] = []
        self.chunks: list[CodeChunk] = []
        self.ground_truth: list[SearchGroundTruth] = []
        self._symbol_index: dict[str, CodeSymbol] = {}
        self._file_content_cache: dict[str, str] = {}

    def collect_all_data(self) -> dict[str, Any]:
        """
        Collect all test data from the source codebase.

        Returns:
            Dictionary containing symbols, chunks, and ground truth data
        """
        print("🔍 Collecting test data from workspace-qdrant-mcp codebase...")

        # Collect Python source files
        python_files = self._find_python_files()
        print(f"📁 Found {len(python_files)} Python files")

        # Extract symbols and chunks from each file
        for file_path in python_files:
            self._process_python_file(file_path)

        # Process documentation files
        doc_files = self._find_documentation_files()
        print(f"📝 Found {len(doc_files)} documentation files")

        for file_path in doc_files:
            self._process_documentation_file(file_path)

        # Generate ground truth search cases
        self._generate_ground_truth()

        print(f"✅ Collected {len(self.symbols)} symbols, {len(self.chunks)} chunks")
        print(f"🎯 Generated {len(self.ground_truth)} ground truth test cases")

        return {
            "symbols": [self._symbol_to_dict(s) for s in self.symbols],
            "chunks": [self._chunk_to_dict(c) for c in self.chunks],
            "ground_truth": [
                self._ground_truth_to_dict(gt) for gt in self.ground_truth
            ],
            "metadata": {
                "source_root": str(self.source_root),
                "collection_timestamp": self._get_timestamp(),
                "stats": {
                    "python_files": len(python_files),
                    "doc_files": len(doc_files),
                    "total_symbols": len(self.symbols),
                    "total_chunks": len(self.chunks),
                    "symbol_types": self._get_symbol_type_stats(),
                },
            },
        }

    def _find_python_files(self) -> list[Path]:
        """Find all Python source files in the project."""
        python_files = []

        # Search in src/ directory
        src_dir = self.source_root / "src"
        if src_dir.exists():
            python_files.extend(src_dir.rglob("*.py"))

        # Include root-level Python files
        python_files.extend(self.source_root.glob("*.py"))

        # Filter out __pycache__ and test files for now
        return [
            f
            for f in python_files
            if "__pycache__" not in str(f) and "tests" not in str(f)
        ]

    def _find_documentation_files(self) -> list[Path]:
        """Find documentation files (README, docs, etc.)."""
        doc_files = []

        # Common documentation files
        doc_patterns = ["*.md", "*.rst", "*.txt"]

        for pattern in doc_patterns:
            doc_files.extend(self.source_root.rglob(pattern))

        # Include specific documentation directories
        docs_dir = self.source_root / "docs"
        if docs_dir.exists():
            doc_files.extend(docs_dir.rglob("*"))

        return [f for f in doc_files if f.is_file()]

    def _process_python_file(self, file_path: Path) -> None:
        """Process a Python file to extract symbols and chunks."""
        try:
            content = file_path.read_text(encoding="utf-8")
            self._file_content_cache[str(file_path)] = content

            # Parse AST
            tree = ast.parse(content)

            # Extract symbols using AST visitor
            visitor = PythonSymbolVisitor(str(file_path), content)
            visitor.visit(tree)

            # Add symbols to collection
            self.symbols.extend(visitor.symbols)

            # Update symbol index
            for symbol in visitor.symbols:
                self._symbol_index[symbol.name] = symbol

            # Create code chunks
            self._create_code_chunks(file_path, content, visitor.symbols)

        except Exception as e:
            print(f"⚠️  Error processing {file_path}: {e}")

    def _process_documentation_file(self, file_path: Path) -> None:
        """Process a documentation file to extract content chunks."""
        try:
            content = file_path.read_text(encoding="utf-8")
            self._file_content_cache[str(file_path)] = content

            # Create documentation chunks
            lines = content.split("\n")
            current_chunk = []
            chunk_start = 1

            for i, line in enumerate(lines, 1):
                current_chunk.append(line)

                # Create chunk at section boundaries or every 10 lines
                if (line.startswith("#") and len(current_chunk) > 1) or len(
                    current_chunk
                ) >= 10:
                    if len(current_chunk) > 1:
                        chunk_content = "\n".join(
                            current_chunk[:-1]
                            if line.startswith("#")
                            else current_chunk
                        )

                        if chunk_content.strip():
                            self.chunks.append(
                                CodeChunk(
                                    content=chunk_content,
                                    file_path=str(file_path),
                                    chunk_type="documentation",
                                    line_start=chunk_start,
                                    line_end=i - (1 if line.startswith("#") else 0),
                                    metadata={
                                        "file_type": file_path.suffix,
                                        "relative_path": str(
                                            file_path.relative_to(self.source_root)
                                        ),
                                    },
                                )
                            )

                    current_chunk = [line] if line.startswith("#") else []
                    chunk_start = i

            # Add final chunk
            if current_chunk and len(current_chunk) > 0:
                chunk_content = "\n".join(current_chunk)
                if chunk_content.strip():
                    self.chunks.append(
                        CodeChunk(
                            content=chunk_content,
                            file_path=str(file_path),
                            chunk_type="documentation",
                            line_start=chunk_start,
                            line_end=len(lines),
                            metadata={
                                "file_type": file_path.suffix,
                                "relative_path": str(
                                    file_path.relative_to(self.source_root)
                                ),
                            },
                        )
                    )

        except Exception as e:
            print(f"⚠️  Error processing documentation {file_path}: {e}")

    def _create_code_chunks(
        self, file_path: Path, content: str, symbols: list[CodeSymbol]
    ) -> None:
        """Create code chunks from Python source."""
        lines = content.split("\n")

        # Create chunks around each symbol
        for symbol in symbols:
            start_line = max(1, symbol.line_number - 2)

            # Find end of symbol (rough heuristic)
            end_line = min(len(lines), symbol.line_number + 10)
            if symbol.source_code:
                source_lines = symbol.source_code.count("\n")
                end_line = symbol.line_number + source_lines

            chunk_lines = lines[start_line - 1 : end_line]
            chunk_content = "\n".join(chunk_lines)

            if chunk_content.strip():
                self.chunks.append(
                    CodeChunk(
                        content=chunk_content,
                        file_path=str(file_path),
                        chunk_type="code",
                        line_start=start_line,
                        line_end=end_line,
                        symbols=[symbol.name],
                        metadata={
                            "symbol_type": symbol.type,
                            "symbol_name": symbol.name,
                            "has_docstring": bool(symbol.docstring),
                            "relative_path": str(
                                file_path.relative_to(self.source_root)
                            ),
                        },
                    )
                )

        # Create additional chunks for module-level docstrings and imports
        if lines:
            # Module docstring chunk
            if lines[0].strip().startswith('"""') or lines[0].strip().startswith("'''"):
                docstring_end = self._find_docstring_end(lines, 0)
                if docstring_end > 0:
                    docstring_content = "\n".join(lines[: docstring_end + 1])
                    self.chunks.append(
                        CodeChunk(
                            content=docstring_content,
                            file_path=str(file_path),
                            chunk_type="docstring",
                            line_start=1,
                            line_end=docstring_end + 1,
                            metadata={
                                "docstring_type": "module",
                                "relative_path": str(
                                    file_path.relative_to(self.source_root)
                                ),
                            },
                        )
                    )

    def _find_docstring_end(self, lines: list[str], start: int) -> int:
        """Find the end of a docstring starting at given line."""
        quote_type = '"""' if '"""' in lines[start] else "'''"
        quote_count = 0

        for i, line in enumerate(lines[start:], start):
            quote_count += line.count(quote_type)
            if quote_count >= 2:  # Opening and closing quotes
                return i

        return start + 5  # Fallback

    def _generate_ground_truth(self) -> None:
        """Generate ground truth search test cases."""
        # Symbol-based searches
        for symbol in self.symbols[:20]:  # Limit for testing
            self.ground_truth.append(
                SearchGroundTruth(
                    query=symbol.name,
                    query_type="symbol",
                    expected_results=[
                        self._generate_chunk_id(c)
                        for c in self.chunks
                        if symbol.name in c.symbols
                    ],
                    expected_symbols=[symbol.name],
                )
            )

        # Semantic searches based on common terms
        semantic_queries = [
            ("FastMCP server", "semantic", "server", "fastmcp"),
            ("Qdrant client connection", "semantic", "client", "qdrant"),
            ("embedding vectors", "semantic", "embedding", "vector"),
            ("collection management", "semantic", "collection", "manager"),
            ("search functionality", "semantic", "search", "query"),
            ("configuration settings", "semantic", "config", "setting"),
        ]

        for query, query_type, *keywords in semantic_queries:
            relevant_chunks = []
            for chunk in self.chunks:
                content_lower = chunk.content.lower()
                if any(keyword in content_lower for keyword in keywords):
                    relevant_chunks.append(self._generate_chunk_id(chunk))

            if relevant_chunks:
                self.ground_truth.append(
                    SearchGroundTruth(
                        query=query,
                        query_type=query_type,
                        expected_results=relevant_chunks[:10],  # Top 10
                    )
                )

        # Exact match searches
        for symbol in self.symbols[::5]:  # Every 5th symbol
            if symbol.docstring and len(symbol.docstring) > 20:
                # Use part of docstring as exact query
                docstring_words = symbol.docstring.split()[:5]
                query = " ".join(docstring_words)

                self.ground_truth.append(
                    SearchGroundTruth(
                        query=query,
                        query_type="exact",
                        expected_results=[
                            self._generate_chunk_id(c)
                            for c in self.chunks
                            if query.lower() in c.content.lower()
                        ],
                        expected_symbols=[symbol.name],
                    )
                )

    def _generate_chunk_id(self, chunk: CodeChunk) -> str:
        """Generate a unique ID for a chunk."""
        content_hash = hashlib.md5(chunk.content.encode()).hexdigest()[:8]
        return f"{Path(chunk.file_path).stem}_{chunk.line_start}_{content_hash}"

    def _get_timestamp(self) -> str:
        """Get current timestamp for metadata."""
        from datetime import datetime

        return datetime.now().isoformat()

    def _get_symbol_type_stats(self) -> dict[str, int]:
        """Get statistics on symbol types."""
        stats = {}
        for symbol in self.symbols:
            stats[symbol.type] = stats.get(symbol.type, 0) + 1
        return stats

    def _symbol_to_dict(self, symbol: CodeSymbol) -> dict:
        """Convert symbol to dictionary."""
        return {
            "name": symbol.name,
            "type": symbol.type,
            "file_path": symbol.file_path,
            "line_number": symbol.line_number,
            "docstring": symbol.docstring,
            "signature": symbol.signature,
            "source_code": symbol.source_code,
            "parent_class": symbol.parent_class,
            "metadata": symbol.metadata,
        }

    def _chunk_to_dict(self, chunk: CodeChunk) -> dict:
        """Convert chunk to dictionary."""
        return {
            "id": self._generate_chunk_id(chunk),
            "content": chunk.content,
            "file_path": chunk.file_path,
            "chunk_type": chunk.chunk_type,
            "line_start": chunk.line_start,
            "line_end": chunk.line_end,
            "symbols": chunk.symbols,
            "metadata": chunk.metadata,
        }

    def _ground_truth_to_dict(self, gt: SearchGroundTruth) -> dict:
        """Convert ground truth to dictionary."""
        return {
            "query": gt.query,
            "query_type": gt.query_type,
            "expected_results": gt.expected_results,
            "expected_symbols": gt.expected_symbols,
            "relevance_scores": gt.relevance_scores,
        }


class PythonSymbolVisitor(ast.NodeVisitor):
    """AST visitor to extract symbols from Python code."""

    def __init__(self, file_path: str, source_code: str):
        self.file_path = file_path
        self.source_lines = source_code.split("\n")
        self.symbols: list[CodeSymbol] = []
        self.current_class: Optional[str] = None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition."""
        previous_class = self.current_class
        self.current_class = node.name

        # Extract class info
        symbol = CodeSymbol(
            name=node.name,
            type="class",
            file_path=self.file_path,
            line_number=node.lineno,
            docstring=ast.get_docstring(node),
            signature=self._get_class_signature(node),
            source_code=self._get_node_source(node),
        )

        self.symbols.append(symbol)

        # Visit class methods
        self.generic_visit(node)
        self.current_class = previous_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition."""
        symbol_type = "method" if self.current_class else "function"

        symbol = CodeSymbol(
            name=node.name,
            type=symbol_type,
            file_path=self.file_path,
            line_number=node.lineno,
            docstring=ast.get_docstring(node),
            signature=self._get_function_signature(node),
            source_code=self._get_node_source(node),
            parent_class=self.current_class,
        )

        self.symbols.append(symbol)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definition."""
        symbol_type = "async_method" if self.current_class else "async_function"

        symbol = CodeSymbol(
            name=node.name,
            type=symbol_type,
            file_path=self.file_path,
            line_number=node.lineno,
            docstring=ast.get_docstring(node),
            signature=f"async {self._get_function_signature(node)}",
            source_code=self._get_node_source(node),
            parent_class=self.current_class,
        )

        self.symbols.append(symbol)
        self.generic_visit(node)

    def _get_function_signature(self, node) -> str:
        """Extract function signature as string."""
        args = []

        # Regular arguments
        for arg in node.args.args:
            args.append(arg.arg)

        # Default arguments
        if node.args.defaults:
            default_start = len(node.args.args) - len(node.args.defaults)
            for i, default in enumerate(node.args.defaults):
                args[default_start + i] += f"={ast.unparse(default)}"

        # *args
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")

        # **kwargs
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")

        return f"{node.name}({', '.join(args)})"

    def _get_class_signature(self, node: ast.ClassDef) -> str:
        """Extract class signature with base classes."""
        bases = [ast.unparse(base) for base in node.bases]
        if bases:
            return f"class {node.name}({', '.join(bases)})"
        return f"class {node.name}"

    def _get_node_source(self, node) -> str:
        """Extract source code for a node."""
        try:
            if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                start_line = node.lineno - 1
                end_line = node.end_lineno if node.end_lineno else start_line + 1
                return "\n".join(self.source_lines[start_line:end_line])
        except Exception:
            pass
        return ""
