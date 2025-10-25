"""
LSP server communication mocking for testing language server interactions.

Provides comprehensive mocking for LSP server operations including initialization,
metadata extraction, symbol resolution, and various LSP-related error scenarios.
"""

import asyncio
import json
import random
from typing import Any, Optional, Union
from unittest.mock import AsyncMock, Mock

from .error_injection import ErrorInjector, FailureScenarios


class LSPErrorInjector(ErrorInjector):
    """Specialized error injector for LSP operations."""

    def __init__(self):
        super().__init__()
        self.failure_modes = {
            "server_not_found": {"probability": 0.0, "error": "LSP server executable not found"},
            "initialization_failed": {"probability": 0.0, "error": "LSP server initialization failed"},
            "protocol_error": {"probability": 0.0, "error": "LSP protocol error"},
            "timeout": {"probability": 0.0, "timeout_seconds": 30.0},
            "invalid_response": {"probability": 0.0, "error": "Invalid LSP response format"},
            "server_crash": {"probability": 0.0, "error": "LSP server crashed"},
            "workspace_error": {"probability": 0.0, "error": "Workspace initialization failed"},
            "file_not_supported": {"probability": 0.0, "error": "File type not supported by LSP"},
            "parsing_error": {"probability": 0.0, "error": "File parsing failed"},
            "memory_exhausted": {"probability": 0.0, "error": "LSP server out of memory"},
        }

    def configure_server_issues(self, probability: float = 0.1):
        """Configure LSP server-related failures."""
        self.failure_modes["server_not_found"]["probability"] = probability
        self.failure_modes["initialization_failed"]["probability"] = probability / 2
        self.failure_modes["server_crash"]["probability"] = probability / 4

    def configure_protocol_issues(self, probability: float = 0.1):
        """Configure LSP protocol-related failures."""
        self.failure_modes["protocol_error"]["probability"] = probability
        self.failure_modes["invalid_response"]["probability"] = probability / 2
        self.failure_modes["timeout"]["probability"] = probability / 3

    def configure_workspace_issues(self, probability: float = 0.1):
        """Configure workspace-related failures."""
        self.failure_modes["workspace_error"]["probability"] = probability
        self.failure_modes["file_not_supported"]["probability"] = probability / 2
        self.failure_modes["parsing_error"]["probability"] = probability / 3


class LSPServerMock:
    """Mock LSP server for testing language server interactions."""

    def __init__(self, language: str = "python", error_injector: LSPErrorInjector | None = None):
        self.language = language
        self.error_injector = error_injector or LSPErrorInjector()
        self.operation_history: list[dict[str, Any]] = []
        self.initialized = False
        self.workspace_folders: list[str] = []
        self.open_documents: dict[str, dict[str, Any]] = {}

        # Setup method mocks
        self._setup_lsp_methods()

    def _setup_lsp_methods(self):
        """Setup LSP method mocks."""
        self.initialize = AsyncMock(side_effect=self._mock_initialize)
        self.shutdown = AsyncMock(side_effect=self._mock_shutdown)
        self.open_document = AsyncMock(side_effect=self._mock_open_document)
        self.close_document = AsyncMock(side_effect=self._mock_close_document)
        self.get_symbols = AsyncMock(side_effect=self._mock_get_symbols)
        self.get_hover_info = AsyncMock(side_effect=self._mock_get_hover_info)
        self.get_definition = AsyncMock(side_effect=self._mock_get_definition)
        self.get_references = AsyncMock(side_effect=self._mock_get_references)
        self.get_diagnostics = AsyncMock(side_effect=self._mock_get_diagnostics)

    async def _inject_lsp_error(self, operation: str) -> None:
        """Inject LSP errors based on configuration."""
        if self.error_injector.should_inject_error():
            error_type = self.error_injector.get_random_error()
            await self._raise_lsp_error(error_type)

    async def _raise_lsp_error(self, error_type: str) -> None:
        """Raise appropriate LSP error based on error type."""
        error_config = self.error_injector.failure_modes.get(error_type, {})

        if error_type == "server_not_found":
            raise FileNotFoundError("LSP server executable not found")
        elif error_type == "initialization_failed":
            raise RuntimeError("LSP server initialization failed")
        elif error_type == "protocol_error":
            raise ValueError("LSP protocol error")
        elif error_type == "timeout":
            timeout = error_config.get("timeout_seconds", 30.0)
            await asyncio.sleep(timeout)
            raise TimeoutError("LSP operation timeout")
        elif error_type == "invalid_response":
            raise ValueError("Invalid LSP response format")
        elif error_type == "server_crash":
            raise ConnectionError("LSP server crashed")
        elif error_type == "workspace_error":
            raise RuntimeError("Workspace initialization failed")
        elif error_type == "file_not_supported":
            raise ValueError("File type not supported by LSP")
        elif error_type == "parsing_error":
            raise SyntaxError("File parsing failed")
        elif error_type == "memory_exhausted":
            raise MemoryError("LSP server out of memory")

    async def _mock_initialize(self, workspace_folders: list[str], capabilities: dict[str, Any]) -> dict[str, Any]:
        """Mock LSP server initialization."""
        await self._inject_lsp_error("initialize")

        self.initialized = True
        self.workspace_folders = workspace_folders

        self.operation_history.append({
            "operation": "initialize",
            "workspace_folders": workspace_folders,
            "capabilities": capabilities
        })

        return {
            "capabilities": {
                "textDocumentSync": {"change": 2, "openClose": True, "save": True},
                "hoverProvider": True,
                "definitionProvider": True,
                "referencesProvider": True,
                "documentSymbolProvider": True,
                "workspaceSymbolProvider": True,
                "codeActionProvider": True,
                "completionProvider": {"triggerCharacters": ["."]},
                "diagnosticProvider": {"interFileDependencies": True}
            },
            "serverInfo": {
                "name": f"Mock {self.language.title()} LSP Server",
                "version": "1.0.0"
            }
        }

    async def _mock_shutdown(self) -> None:
        """Mock LSP server shutdown."""
        self.initialized = False
        self.workspace_folders.clear()
        self.open_documents.clear()

        self.operation_history.append({
            "operation": "shutdown"
        })

    async def _mock_open_document(self, file_path: str, content: str, language_id: str) -> None:
        """Mock opening a document in LSP."""
        await self._inject_lsp_error("open_document")

        if not self.initialized:
            raise RuntimeError("LSP server not initialized")

        self.open_documents[file_path] = {
            "content": content,
            "language_id": language_id,
            "version": 1,
            "opened_at": "2024-01-01T12:00:00Z"
        }

        self.operation_history.append({
            "operation": "open_document",
            "file_path": file_path,
            "language_id": language_id,
            "content_length": len(content)
        })

    async def _mock_close_document(self, file_path: str) -> None:
        """Mock closing a document in LSP."""
        if file_path in self.open_documents:
            del self.open_documents[file_path]

        self.operation_history.append({
            "operation": "close_document",
            "file_path": file_path
        })

    async def _mock_get_symbols(self, file_path: str) -> list[dict[str, Any]]:
        """Mock getting document symbols."""
        await self._inject_lsp_error("get_symbols")

        if not self.initialized:
            raise RuntimeError("LSP server not initialized")

        if file_path not in self.open_documents:
            raise ValueError(f"Document not open: {file_path}")

        self.operation_history.append({
            "operation": "get_symbols",
            "file_path": file_path
        })

        # Generate realistic symbols based on language
        return self._generate_mock_symbols(file_path)

    async def _mock_get_hover_info(self, file_path: str, line: int, column: int) -> dict[str, Any] | None:
        """Mock getting hover information."""
        await self._inject_lsp_error("get_hover_info")

        if not self.initialized:
            raise RuntimeError("LSP server not initialized")

        self.operation_history.append({
            "operation": "get_hover_info",
            "file_path": file_path,
            "line": line,
            "column": column
        })

        return {
            "contents": {
                "kind": "markdown",
                "value": f"Hover information for position {line}:{column} in {file_path}"
            },
            "range": {
                "start": {"line": line, "character": column},
                "end": {"line": line, "character": column + 10}
            }
        }

    async def _mock_get_definition(self, file_path: str, line: int, column: int) -> list[dict[str, Any]]:
        """Mock getting symbol definition."""
        await self._inject_lsp_error("get_definition")

        if not self.initialized:
            raise RuntimeError("LSP server not initialized")

        self.operation_history.append({
            "operation": "get_definition",
            "file_path": file_path,
            "line": line,
            "column": column
        })

        return [{
            "uri": file_path,
            "range": {
                "start": {"line": max(0, line - 5), "character": 0},
                "end": {"line": max(0, line - 5), "character": 20}
            }
        }]

    async def _mock_get_references(self, file_path: str, line: int, column: int, include_declaration: bool = True) -> list[dict[str, Any]]:
        """Mock getting symbol references."""
        await self._inject_lsp_error("get_references")

        if not self.initialized:
            raise RuntimeError("LSP server not initialized")

        self.operation_history.append({
            "operation": "get_references",
            "file_path": file_path,
            "line": line,
            "column": column,
            "include_declaration": include_declaration
        })

        # Generate mock references
        references = []
        for i in range(random.randint(1, 5)):
            references.append({
                "uri": file_path if i == 0 else f"other_file_{i}.py",
                "range": {
                    "start": {"line": line + i, "character": column},
                    "end": {"line": line + i, "character": column + 10}
                }
            })

        return references

    async def _mock_get_diagnostics(self, file_path: str) -> list[dict[str, Any]]:
        """Mock getting file diagnostics."""
        await self._inject_lsp_error("get_diagnostics")

        if not self.initialized:
            raise RuntimeError("LSP server not initialized")

        self.operation_history.append({
            "operation": "get_diagnostics",
            "file_path": file_path
        })

        # Generate mock diagnostics
        diagnostics = []
        for i in range(random.randint(0, 3)):
            diagnostics.append({
                "range": {
                    "start": {"line": i * 5, "character": 0},
                    "end": {"line": i * 5, "character": 20}
                },
                "severity": random.choice([1, 2, 3, 4]),  # Error, Warning, Information, Hint
                "code": f"E{100 + i}",
                "source": f"{self.language}-lsp",
                "message": f"Mock diagnostic message {i}"
            })

        return diagnostics

    def _generate_mock_symbols(self, file_path: str) -> list[dict[str, Any]]:
        """Generate realistic symbols based on language."""
        symbols = []

        if self.language == "python":
            symbols.extend([
                {
                    "name": "MockClass",
                    "kind": 5,  # Class
                    "range": {"start": {"line": 0, "character": 0}, "end": {"line": 10, "character": 0}},
                    "selectionRange": {"start": {"line": 0, "character": 6}, "end": {"line": 0, "character": 15}},
                    "children": [
                        {
                            "name": "__init__",
                            "kind": 6,  # Method
                            "range": {"start": {"line": 1, "character": 4}, "end": {"line": 3, "character": 0}},
                            "selectionRange": {"start": {"line": 1, "character": 8}, "end": {"line": 1, "character": 16}}
                        },
                        {
                            "name": "mock_method",
                            "kind": 6,  # Method
                            "range": {"start": {"line": 4, "character": 4}, "end": {"line": 8, "character": 0}},
                            "selectionRange": {"start": {"line": 4, "character": 8}, "end": {"line": 4, "character": 19}}
                        }
                    ]
                },
                {
                    "name": "mock_function",
                    "kind": 12,  # Function
                    "range": {"start": {"line": 12, "character": 0}, "end": {"line": 15, "character": 0}},
                    "selectionRange": {"start": {"line": 12, "character": 4}, "end": {"line": 12, "character": 17}}
                }
            ])
        elif self.language == "javascript":
            symbols.extend([
                {
                    "name": "MockFunction",
                    "kind": 12,  # Function
                    "range": {"start": {"line": 0, "character": 0}, "end": {"line": 5, "character": 1}},
                    "selectionRange": {"start": {"line": 0, "character": 9}, "end": {"line": 0, "character": 21}}
                },
                {
                    "name": "mockVariable",
                    "kind": 13,  # Variable
                    "range": {"start": {"line": 7, "character": 0}, "end": {"line": 7, "character": 25}},
                    "selectionRange": {"start": {"line": 7, "character": 6}, "end": {"line": 7, "character": 18}}
                }
            ])

        return symbols

    def get_operation_history(self) -> list[dict[str, Any]]:
        """Get history of LSP operations."""
        return self.operation_history.copy()

    def reset_state(self) -> None:
        """Reset LSP server state."""
        self.operation_history.clear()
        self.initialized = False
        self.workspace_folders.clear()
        self.open_documents.clear()
        self.error_injector.reset()


class LSPMetadataExtractorMock:
    """Mock LSP metadata extractor for testing code analysis."""

    def __init__(self, error_injector: LSPErrorInjector | None = None):
        self.error_injector = error_injector or LSPErrorInjector()
        self.operation_history: list[dict[str, Any]] = []
        self.language_servers: dict[str, LSPServerMock] = {}

        # Setup method mocks
        self._setup_extractor_methods()

    def _setup_extractor_methods(self):
        """Setup metadata extractor method mocks."""
        self.extract_metadata = AsyncMock(side_effect=self._mock_extract_metadata)
        self.extract_symbols = AsyncMock(side_effect=self._mock_extract_symbols)
        self.extract_dependencies = AsyncMock(side_effect=self._mock_extract_dependencies)
        self.extract_documentation = AsyncMock(side_effect=self._mock_extract_documentation)
        self.get_supported_languages = Mock(side_effect=self._mock_get_supported_languages)

    async def _mock_extract_metadata(self, file_path: str, content: str, language: str) -> dict[str, Any]:
        """Mock comprehensive metadata extraction."""
        await self._inject_extractor_error("extract_metadata")

        self.operation_history.append({
            "operation": "extract_metadata",
            "file_path": file_path,
            "language": language,
            "content_length": len(content)
        })

        # Get or create LSP server for language
        if language not in self.language_servers:
            self.language_servers[language] = LSPServerMock(language, self.error_injector)

        lsp_server = self.language_servers[language]

        # Initialize if needed
        if not lsp_server.initialized:
            await lsp_server.initialize(["/mock/workspace"], {})

        # Open document
        await lsp_server.open_document(file_path, content, language)

        # Extract comprehensive metadata
        symbols = await lsp_server.get_symbols(file_path)
        diagnostics = await lsp_server.get_diagnostics(file_path)

        return {
            "file_path": file_path,
            "language": language,
            "symbols": symbols,
            "diagnostics": diagnostics,
            "metrics": {
                "lines_of_code": len(content.splitlines()),
                "symbol_count": len(symbols),
                "diagnostic_count": len(diagnostics),
                "complexity_score": random.randint(1, 10)
            },
            "dependencies": self._extract_mock_dependencies(content, language),
            "documentation": self._extract_mock_documentation(content, language)
        }

    async def _mock_extract_symbols(self, file_path: str, content: str, language: str) -> list[dict[str, Any]]:
        """Mock symbol extraction only."""
        await self._inject_extractor_error("extract_symbols")

        self.operation_history.append({
            "operation": "extract_symbols",
            "file_path": file_path,
            "language": language
        })

        # Get LSP server and extract symbols
        if language not in self.language_servers:
            self.language_servers[language] = LSPServerMock(language, self.error_injector)

        lsp_server = self.language_servers[language]
        if not lsp_server.initialized:
            await lsp_server.initialize(["/mock/workspace"], {})

        await lsp_server.open_document(file_path, content, language)
        return await lsp_server.get_symbols(file_path)

    async def _mock_extract_dependencies(self, file_path: str, content: str, language: str) -> list[dict[str, Any]]:
        """Mock dependency extraction."""
        await self._inject_extractor_error("extract_dependencies")

        self.operation_history.append({
            "operation": "extract_dependencies",
            "file_path": file_path,
            "language": language
        })

        return self._extract_mock_dependencies(content, language)

    async def _mock_extract_documentation(self, file_path: str, content: str, language: str) -> dict[str, Any]:
        """Mock documentation extraction."""
        await self._inject_extractor_error("extract_documentation")

        self.operation_history.append({
            "operation": "extract_documentation",
            "file_path": file_path,
            "language": language
        })

        return self._extract_mock_documentation(content, language)

    def _mock_get_supported_languages(self) -> list[str]:
        """Mock getting supported languages."""
        return [
            "python", "javascript", "typescript", "java", "csharp", "cpp", "c",
            "go", "rust", "php", "ruby", "kotlin", "swift", "scala", "r"
        ]

    async def _inject_extractor_error(self, operation: str) -> None:
        """Inject extractor-specific errors."""
        if self.error_injector.should_inject_error():
            error_type = self.error_injector.get_random_error()
            if error_type in ["parsing_error", "file_not_supported"]:
                await self.error_injector._raise_lsp_error(error_type)

    def _extract_mock_dependencies(self, content: str, language: str) -> list[dict[str, Any]]:
        """Extract mock dependencies based on language."""
        dependencies = []

        if language == "python":
            # Look for import statements (mock)
            import_keywords = ["import", "from"]
            for keyword in import_keywords:
                if keyword in content:
                    dependencies.append({
                        "name": f"mock_{keyword}_dependency",
                        "type": "import",
                        "line": random.randint(1, 10),
                        "source": "external"
                    })

        elif language == "javascript":
            # Look for require/import statements (mock)
            if "require(" in content or "import" in content:
                dependencies.append({
                    "name": "mock_js_dependency",
                    "type": "module",
                    "line": random.randint(1, 5),
                    "source": "npm"
                })

        return dependencies

    def _extract_mock_documentation(self, content: str, language: str) -> dict[str, Any]:
        """Extract mock documentation based on language."""
        doc_info = {
            "docstring_count": 0,
            "comment_count": 0,
            "documentation_coverage": 0.0
        }

        if language == "python":
            # Mock Python docstring detection
            doc_info["docstring_count"] = content.count('"""') // 2
            doc_info["comment_count"] = content.count('#')

        elif language == "javascript":
            # Mock JavaScript comment detection
            doc_info["comment_count"] = content.count('//') + content.count('/*')

        # Calculate mock coverage
        total_symbols = max(1, content.count('def ') + content.count('function ') + content.count('class '))
        doc_info["documentation_coverage"] = min(1.0, doc_info["docstring_count"] / total_symbols)

        return doc_info

    def reset_state(self) -> None:
        """Reset extractor state."""
        self.operation_history.clear()
        for server in self.language_servers.values():
            server.reset_state()
        self.language_servers.clear()
        self.error_injector.reset()


class LanguageDetectorMock:
    """Mock language detector for file type identification."""

    def __init__(self, error_injector: LSPErrorInjector | None = None):
        self.error_injector = error_injector or LSPErrorInjector()
        self.operation_history: list[dict[str, Any]] = []

        # Setup method mocks
        self.detect_language = Mock(side_effect=self._mock_detect_language)
        self.get_lsp_server_for_language = Mock(side_effect=self._mock_get_lsp_server_for_language)
        self.is_supported = Mock(side_effect=self._mock_is_supported)

    def _mock_detect_language(self, file_path: str, content: str | None = None) -> str:
        """Mock language detection from file path or content."""
        self.operation_history.append({
            "operation": "detect_language",
            "file_path": file_path,
            "has_content": content is not None
        })

        # Simple extension-based detection
        if file_path.endswith('.py'):
            return "python"
        elif file_path.endswith(('.js', '.jsx')):
            return "javascript"
        elif file_path.endswith(('.ts', '.tsx')):
            return "typescript"
        elif file_path.endswith('.java'):
            return "java"
        elif file_path.endswith('.go'):
            return "go"
        elif file_path.endswith('.rs'):
            return "rust"
        elif file_path.endswith(('.cpp', '.cc', '.cxx')):
            return "cpp"
        elif file_path.endswith('.c'):
            return "c"
        elif file_path.endswith('.cs'):
            return "csharp"
        elif file_path.endswith('.php'):
            return "php"
        elif file_path.endswith('.rb'):
            return "ruby"
        else:
            return "plaintext"

    def _mock_get_lsp_server_for_language(self, language: str) -> str | None:
        """Mock getting LSP server executable for language."""
        self.operation_history.append({
            "operation": "get_lsp_server_for_language",
            "language": language
        })

        server_map = {
            "python": "pylsp",
            "javascript": "typescript-language-server",
            "typescript": "typescript-language-server",
            "java": "jdtls",
            "go": "gopls",
            "rust": "rust-analyzer",
            "cpp": "clangd",
            "c": "clangd",
            "csharp": "omnisharp",
            "php": "intelephense",
            "ruby": "solargraph"
        }

        return server_map.get(language)

    def _mock_is_supported(self, language: str) -> bool:
        """Mock checking if language is supported."""
        supported_languages = {
            "python", "javascript", "typescript", "java", "go", "rust",
            "cpp", "c", "csharp", "php", "ruby", "kotlin", "swift"
        }
        return language in supported_languages

    def reset_state(self) -> None:
        """Reset detector state."""
        self.operation_history.clear()


def create_lsp_mock(
    component: str = "server",
    language: str = "python",
    with_error_injection: bool = False,
    error_probability: float = 0.1
) -> LSPServerMock | LSPMetadataExtractorMock | LanguageDetectorMock:
    """
    Create an LSP mock component with optional error injection.

    Args:
        component: Type of mock ("server", "extractor", "detector")
        language: Programming language for server mock
        with_error_injection: Enable error injection
        error_probability: Probability of errors (0.0 to 1.0)

    Returns:
        Configured LSP mock instance
    """
    error_injector = None
    if with_error_injection:
        error_injector = LSPErrorInjector()
        error_injector.configure_server_issues(error_probability)
        error_injector.configure_protocol_issues(error_probability)
        error_injector.configure_workspace_issues(error_probability)

    if component == "server":
        return LSPServerMock(language, error_injector)
    elif component == "extractor":
        return LSPMetadataExtractorMock(error_injector)
    elif component == "detector":
        return LanguageDetectorMock(error_injector)
    else:
        raise ValueError(f"Unknown component type: {component}")


# Convenience functions for common scenarios
def create_basic_lsp_server(language: str = "python") -> LSPServerMock:
    """Create basic LSP server mock without error injection."""
    return create_lsp_mock("server", language)


def create_failing_lsp_server(language: str = "python", error_rate: float = 0.3) -> LSPServerMock:
    """Create LSP server mock with high failure rate."""
    return create_lsp_mock("server", language, with_error_injection=True, error_probability=error_rate)


def create_realistic_metadata_extractor() -> LSPMetadataExtractorMock:
    """Create metadata extractor mock with realistic error rates."""
    return create_lsp_mock("extractor", with_error_injection=True, error_probability=0.05)
