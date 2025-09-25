"""FastAPI-based interactive documentation server."""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

try:
    from ..generators.ast_parser import PythonASTParser, DocumentationNode
    from ..generators.template_engine import DocumentationTemplateEngine
    from ..validation.coverage_analyzer import DocumentationCoverageAnalyzer
    from ..validation.quality_checker import DocumentationQualityChecker
    from .sandbox import CodeSandbox
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from generators.ast_parser import PythonASTParser, DocumentationNode
    from generators.template_engine import DocumentationTemplateEngine
    from validation.coverage_analyzer import DocumentationCoverageAnalyzer
    from validation.quality_checker import DocumentationQualityChecker
    from server.sandbox import CodeSandbox


class CodeExecutionRequest(BaseModel):
    """Request model for code execution."""
    code: str
    language: str = "python"
    context: Optional[Dict[str, Any]] = None


class CodeExecutionResponse(BaseModel):
    """Response model for code execution."""
    success: bool
    result: Optional[str] = None
    error: Optional[str] = None
    output: Optional[str] = None
    execution_time: Optional[float] = None


class DocumentationSearchRequest(BaseModel):
    """Request model for documentation search."""
    query: str
    member_types: Optional[List[str]] = None
    include_private: bool = False


class DocumentationApp:
    """Interactive documentation application."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the documentation app.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.app = FastAPI(
            title="Interactive Documentation",
            description="Live documentation with executable examples",
            version="1.0.0"
        )

        # Initialize components
        self.ast_parser = PythonASTParser(include_private=False)
        self.template_engine = None
        self.coverage_analyzer = DocumentationCoverageAnalyzer()
        self.quality_checker = DocumentationQualityChecker()
        self.sandbox = CodeSandbox(
            timeout=config.get('sandbox', {}).get('timeout', 30),
            memory_limit=config.get('sandbox', {}).get('memory_limit', 128)
        )

        # Documentation data
        self.modules: List[DocumentationNode] = []
        self.search_index: Dict[str, Any] = {}

        # Set up templates if template directory exists
        template_dir = config.get('templates_dir', 'docs/templates')
        if os.path.exists(template_dir):
            self.templates = Jinja2Templates(directory=template_dir)
            self.template_engine = DocumentationTemplateEngine(template_dir, config)

        # Set up routes
        self._setup_routes()

        # Load documentation
        self._load_documentation()

    def _setup_routes(self):
        """Set up API routes."""

        @self.app.get("/", response_class=HTMLResponse)
        async def home(request: Request):
            """Home page with documentation overview."""
            if not self.templates:
                return HTMLResponse("Documentation templates not found")

            return self.templates.TemplateResponse("home.html", {
                "request": request,
                "modules": self.modules,
                "config": self.config
            })

        @self.app.get("/api/modules", response_model=List[Dict])
        async def list_modules():
            """List all available modules."""
            return [
                {
                    "name": module.name,
                    "docstring": module.docstring,
                    "source_file": module.source_file,
                    "children_count": len(module.children)
                }
                for module in self.modules
            ]

        @self.app.get("/api/modules/{module_name}", response_model=Dict)
        async def get_module(module_name: str):
            """Get detailed information about a module."""
            module = self._find_module(module_name)
            if not module:
                raise HTTPException(status_code=404, detail="Module not found")

            return self._serialize_node(module)

        @self.app.get("/api/modules/{module_name}/coverage")
        async def get_module_coverage(module_name: str):
            """Get coverage analysis for a module."""
            module = self._find_module(module_name)
            if not module:
                raise HTTPException(status_code=404, detail="Module not found")

            # Create a temporary file for analysis
            import tempfile
            if module.source_file and os.path.exists(module.source_file):
                coverage = self.coverage_analyzer.analyze_file(module.source_file)
                return {
                    "coverage_percentage": coverage.stats.coverage_percentage,
                    "total_items": coverage.stats.total_items,
                    "documented_items": coverage.stats.documented_items,
                    "missing_docstring": coverage.stats.missing_docstring,
                    "members": [
                        {
                            "name": member.name,
                            "type": member.member_type.value,
                            "has_docstring": member.has_docstring,
                            "issues": member.issues
                        }
                        for member in coverage.members
                    ]
                }
            else:
                return {"error": "Source file not available"}

        @self.app.get("/api/modules/{module_name}/quality")
        async def get_module_quality(module_name: str):
            """Get quality analysis for a module."""
            module = self._find_module(module_name)
            if not module:
                raise HTTPException(status_code=404, detail="Module not found")

            report = self.quality_checker.check_project_quality([module])
            return {
                "overall_score": report.overall_score,
                "summary_stats": report.summary_stats,
                "member_reports": [
                    {
                        "name": r.member_name,
                        "type": r.member_type.value,
                        "score": r.quality_score,
                        "issues": [
                            {
                                "type": i.issue_type.value,
                                "message": i.message,
                                "severity": i.severity,
                                "suggestion": i.suggestion
                            }
                            for i in r.issues
                        ],
                        "strengths": r.strengths
                    }
                    for r in report.member_reports
                ]
            }

        @self.app.post("/api/execute", response_model=CodeExecutionResponse)
        async def execute_code(request: CodeExecutionRequest):
            """Execute code in a sandbox environment."""
            try:
                result = await self.sandbox.execute_code(
                    request.code,
                    language=request.language,
                    context=request.context or {}
                )
                return CodeExecutionResponse(
                    success=True,
                    result=result.get('result'),
                    output=result.get('output'),
                    execution_time=result.get('execution_time')
                )
            except Exception as e:
                return CodeExecutionResponse(
                    success=False,
                    error=str(e)
                )

        @self.app.post("/api/search")
        async def search_documentation(request: DocumentationSearchRequest):
            """Search through documentation."""
            results = self._search_members(
                request.query,
                request.member_types,
                request.include_private
            )
            return {
                "query": request.query,
                "total_results": len(results),
                "results": results
            }

        @self.app.get("/api/examples/{module_name}/{member_name}")
        async def get_member_examples(module_name: str, member_name: str):
            """Get examples for a specific member."""
            module = self._find_module(module_name)
            if not module:
                raise HTTPException(status_code=404, detail="Module not found")

            member = self._find_member_in_node(module, member_name)
            if not member:
                raise HTTPException(status_code=404, detail="Member not found")

            examples = self._extract_examples(member)
            return {
                "member": member_name,
                "examples": examples
            }

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "modules_loaded": len(self.modules),
                "sandbox_available": self.sandbox.is_available()
            }

    def _load_documentation(self):
        """Load documentation from source directories."""
        source_dirs = self.config.get('sources', {}).get('python', [])

        for source_dir in source_dirs:
            if not os.path.exists(source_dir):
                continue

            try:
                modules = self.ast_parser.parse_directory(source_dir, recursive=True)
                self.modules.extend(modules)
            except Exception as e:
                print(f"Warning: Could not load documentation from {source_dir}: {e}")

        # Build search index
        self._build_search_index()

    def _build_search_index(self):
        """Build search index for fast lookups."""
        self.search_index = {}

        for module in self.modules:
            self._index_node(module, module.name)

    def _index_node(self, node: DocumentationNode, module_name: str):
        """Add a node to the search index."""
        # Index by name
        key = f"{module_name}.{node.name}".lower()
        self.search_index[key] = {
            "module": module_name,
            "node": node,
            "full_name": f"{module_name}.{node.name}",
            "type": node.member_type.value
        }

        # Index by docstring content
        if node.docstring:
            words = node.docstring.lower().split()
            for word in words:
                if len(word) > 3:  # Only index meaningful words
                    if word not in self.search_index:
                        self.search_index[word] = []
                    if isinstance(self.search_index[word], list):
                        self.search_index[word].append({
                            "module": module_name,
                            "node": node,
                            "full_name": f"{module_name}.{node.name}",
                            "type": node.member_type.value
                        })

        # Recursively index children
        for child in node.children:
            self._index_node(child, module_name)

    def _find_module(self, module_name: str) -> Optional[DocumentationNode]:
        """Find a module by name."""
        for module in self.modules:
            if module.name == module_name:
                return module
        return None

    def _find_member_in_node(self, node: DocumentationNode, member_name: str) -> Optional[DocumentationNode]:
        """Find a member within a node."""
        if node.name == member_name:
            return node

        for child in node.children:
            result = self._find_member_in_node(child, member_name)
            if result:
                return result

        return None

    def _serialize_node(self, node: DocumentationNode) -> Dict[str, Any]:
        """Serialize a DocumentationNode to dict."""
        return {
            "name": node.name,
            "type": node.member_type.value,
            "docstring": node.docstring,
            "signature": node.signature,
            "source_file": node.source_file,
            "line_number": node.line_number,
            "is_private": node.is_private,
            "parameters": [
                {
                    "name": param.name,
                    "annotation": param.annotation,
                    "default": param.default,
                    "kind": param.kind,
                    "description": param.description
                }
                for param in node.parameters
            ] if node.parameters else [],
            "return_annotation": node.return_annotation,
            "return_description": node.return_description,
            "examples": node.examples,
            "decorators": node.decorators,
            "children": [self._serialize_node(child) for child in node.children]
        }

    def _search_members(self, query: str, member_types: Optional[List[str]] = None,
                       include_private: bool = False) -> List[Dict[str, Any]]:
        """Search for members matching the query."""
        query_lower = query.lower()
        results = []

        # Search in index
        for key, value in self.search_index.items():
            if isinstance(value, list):  # Word-based search
                for item in value:
                    if query_lower in item["full_name"].lower() or query_lower in key:
                        self._add_search_result(results, item, include_private, member_types)
            else:  # Direct match
                if query_lower in key:
                    self._add_search_result(results, value, include_private, member_types)

        # Remove duplicates and limit results
        seen = set()
        unique_results = []
        for result in results:
            key = (result["module"], result["name"])
            if key not in seen:
                seen.add(key)
                unique_results.append(result)

        return unique_results[:50]  # Limit to 50 results

    def _add_search_result(self, results: List[Dict], item: Dict,
                          include_private: bool, member_types: Optional[List[str]]):
        """Add an item to search results if it matches filters."""
        node = item["node"]

        # Filter private members
        if not include_private and node.is_private:
            return

        # Filter by member type
        if member_types and node.member_type.value not in member_types:
            return

        results.append({
            "module": item["module"],
            "name": node.name,
            "full_name": item["full_name"],
            "type": item["type"],
            "docstring": node.docstring[:200] + "..." if node.docstring and len(node.docstring) > 200 else node.docstring,
            "signature": node.signature,
            "source_file": node.source_file,
            "line_number": node.line_number
        })

    def _extract_examples(self, member: DocumentationNode) -> List[Dict[str, str]]:
        """Extract examples from a member's documentation."""
        examples = []

        if member.examples:
            for example in member.examples:
                examples.append({
                    "type": "documented",
                    "code": example,
                    "description": "Example from documentation"
                })

        # Extract code examples from docstring
        if member.docstring:
            import re
            # Find >>> style examples
            python_examples = re.findall(r'>>>\s+(.+?)(?=\n(?!\.\.\.)|$)', member.docstring, re.MULTILINE | re.DOTALL)
            for example in python_examples:
                examples.append({
                    "type": "interactive",
                    "code": example.strip(),
                    "description": "Interactive Python example"
                })

            # Find code blocks
            code_blocks = re.findall(r'```(?:python)?\n(.+?)\n```', member.docstring, re.MULTILINE | re.DOTALL)
            for block in code_blocks:
                examples.append({
                    "type": "code_block",
                    "code": block.strip(),
                    "description": "Code block example"
                })

        return examples


def create_documentation_app(config: Dict[str, Any]) -> FastAPI:
    """Create and configure the documentation application.

    Args:
        config: Configuration dictionary

    Returns:
        Configured FastAPI application
    """
    doc_app = DocumentationApp(config)

    # Mount static files if available
    static_dir = config.get('static_dir', 'docs/static')
    if os.path.exists(static_dir):
        doc_app.app.mount("/static", StaticFiles(directory=static_dir), name="static")

    return doc_app.app


def run_server(config: Dict[str, Any], host: str = "127.0.0.1", port: int = 8080):
    """Run the documentation server.

    Args:
        config: Configuration dictionary
        host: Host to bind to
        port: Port to listen on
    """
    import uvicorn

    app = create_documentation_app(config)

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=config.get('server', {}).get('auto_reload', False)
    )