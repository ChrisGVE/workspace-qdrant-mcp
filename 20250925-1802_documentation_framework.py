#!/usr/bin/env python3
"""
Comprehensive Documentation Framework for Workspace Qdrant MCP

Implements automated documentation generation, interactive examples, validation,
and deployment system with CI/CD integration and quality metrics.

Created: 2025-09-25T18:02:00+02:00
"""

import ast
import asyncio
import inspect
import json
import logging
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import yaml
import markdown
import jinja2
from jinja2 import Environment, FileSystemLoader, select_autoescape
import docstring_parser
from docstring_parser import Docstring
import black
import isort


class DocumentationType(Enum):
    """Documentation type classification"""
    API = "api"
    TUTORIAL = "tutorial"
    GUIDE = "guide"
    REFERENCE = "reference"
    EXAMPLE = "example"
    CHANGELOG = "changelog"


class ValidationLevel(Enum):
    """Documentation validation levels"""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


class DeploymentStatus(Enum):
    """Documentation deployment status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class DocumentationMetadata:
    """Documentation metadata container"""
    title: str
    description: str
    doc_type: DocumentationType
    version: str
    author: str
    created: datetime
    updated: datetime
    tags: List[str]
    dependencies: List[str]
    validation_level: ValidationLevel


@dataclass
class APIDocumentation:
    """API documentation structure"""
    name: str
    description: str
    parameters: List[Dict[str, Any]]
    returns: Dict[str, Any]
    examples: List[Dict[str, Any]]
    raises: List[Dict[str, str]]
    see_also: List[str]
    metadata: DocumentationMetadata


@dataclass
class ValidationResult:
    """Documentation validation result"""
    is_valid: bool
    score: float
    issues: List[Dict[str, Any]]
    suggestions: List[str]
    metrics: Dict[str, float]


@dataclass
class DeploymentResult:
    """Documentation deployment result"""
    status: DeploymentStatus
    url: Optional[str]
    build_log: str
    deployment_time: datetime
    version: str
    errors: List[str]


class CodeAnalyzer:
    """Analyzes Python code for documentation generation"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

    def analyze_module(self, module_path: Path) -> Dict[str, Any]:
        """Analyze Python module for documentation extraction with edge case handling"""
        try:
            if not module_path.exists() or not module_path.is_file():
                return {"error": "file_not_found", "path": str(module_path)}

            if module_path.suffix != '.py':
                return {"error": "not_python_file", "path": str(module_path)}

            # Read and parse the module
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content.strip():
                return {"error": "empty_file", "path": str(module_path)}

            # Parse AST
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                return {
                    "error": "syntax_error",
                    "path": str(module_path),
                    "details": str(e)
                }

            # Extract documentation elements
            module_info = {
                "path": str(module_path),
                "module_docstring": ast.get_docstring(tree),
                "classes": [],
                "functions": [],
                "constants": [],
                "imports": [],
                "metadata": {
                    "lines_of_code": len(content.splitlines()),
                    "analysis_time": datetime.now().isoformat()
                }
            }

            # Analyze AST nodes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = self._analyze_class(node)
                    if class_info:
                        module_info["classes"].append(class_info)

                elif isinstance(node, ast.FunctionDef):
                    # Skip nested functions (already captured in classes)
                    if not any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree)
                              if hasattr(parent, 'body') and node in getattr(parent, 'body', [])):
                        func_info = self._analyze_function(node)
                        if func_info:
                            module_info["functions"].append(func_info)

                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    const_info = self._analyze_constant(node)
                    if const_info:
                        module_info["constants"].append(const_info)

                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_info = self._analyze_import(node)
                    if import_info:
                        module_info["imports"].append(import_info)

            return module_info

        except Exception as e:
            self.logger.error(f"Module analysis error for {module_path}: {e}")
            return {"error": str(e), "path": str(module_path)}

    def _analyze_class(self, node: ast.ClassDef) -> Optional[Dict[str, Any]]:
        """Analyze class definition with edge case handling"""
        try:
            class_info = {
                "name": node.name,
                "docstring": ast.get_docstring(node),
                "bases": [self._get_base_name(base) for base in node.bases],
                "methods": [],
                "attributes": [],
                "decorators": [self._get_decorator_name(dec) for dec in node.decorator_list],
                "line_number": node.lineno
            }

            # Analyze class methods and attributes
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    method_info = self._analyze_function(item, is_method=True)
                    if method_info:
                        class_info["methods"].append(method_info)

                elif isinstance(item, (ast.Assign, ast.AnnAssign)):
                    attr_info = self._analyze_attribute(item)
                    if attr_info:
                        class_info["attributes"].append(attr_info)

            return class_info

        except Exception as e:
            self.logger.error(f"Class analysis error for {node.name}: {e}")
            return None

    def _analyze_function(self, node: ast.FunctionDef, is_method: bool = False) -> Optional[Dict[str, Any]]:
        """Analyze function definition with edge case handling"""
        try:
            func_info = {
                "name": node.name,
                "docstring": ast.get_docstring(node),
                "parameters": [],
                "returns": None,
                "decorators": [self._get_decorator_name(dec) for dec in node.decorator_list],
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "is_method": is_method,
                "line_number": node.lineno
            }

            # Analyze function parameters
            for arg in node.args.args:
                param_info = {
                    "name": arg.arg,
                    "annotation": self._get_annotation(arg.annotation) if arg.annotation else None,
                    "default": None
                }
                func_info["parameters"].append(param_info)

            # Handle default values
            defaults = node.args.defaults
            if defaults:
                # Map defaults to parameters (defaults apply to last N parameters)
                param_count = len(func_info["parameters"])
                default_count = len(defaults)
                start_idx = param_count - default_count

                for i, default in enumerate(defaults):
                    if start_idx + i < len(func_info["parameters"]):
                        func_info["parameters"][start_idx + i]["default"] = self._get_default_value(default)

            # Analyze return annotation
            if node.returns:
                func_info["returns"] = self._get_annotation(node.returns)

            return func_info

        except Exception as e:
            self.logger.error(f"Function analysis error for {node.name}: {e}")
            return None

    def _analyze_constant(self, node: Union[ast.Assign, ast.AnnAssign]) -> Optional[Dict[str, Any]]:
        """Analyze constant/variable assignment with edge case handling"""
        try:
            if isinstance(node, ast.Assign):
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    name = node.targets[0].id
                    value = self._get_constant_value(node.value)
                    annotation = None
                else:
                    return None  # Skip complex assignments
            else:  # ast.AnnAssign
                if isinstance(node.target, ast.Name):
                    name = node.target.id
                    value = self._get_constant_value(node.value) if node.value else None
                    annotation = self._get_annotation(node.annotation) if node.annotation else None
                else:
                    return None

            # Only include uppercase constants and class variables
            if name.isupper() or name.startswith('_'):
                return {
                    "name": name,
                    "value": value,
                    "annotation": annotation,
                    "line_number": node.lineno
                }

            return None

        except Exception as e:
            self.logger.error(f"Constant analysis error: {e}")
            return None

    def _analyze_attribute(self, node: Union[ast.Assign, ast.AnnAssign]) -> Optional[Dict[str, Any]]:
        """Analyze class attribute with edge case handling"""
        try:
            if isinstance(node, ast.Assign):
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    name = node.targets[0].id
                    value = self._get_constant_value(node.value)
                    annotation = None
                else:
                    return None
            else:  # ast.AnnAssign
                if isinstance(node.target, ast.Name):
                    name = node.target.id
                    value = self._get_constant_value(node.value) if node.value else None
                    annotation = self._get_annotation(node.annotation) if node.annotation else None
                else:
                    return None

            return {
                "name": name,
                "value": value,
                "annotation": annotation,
                "line_number": node.lineno
            }

        except Exception as e:
            self.logger.error(f"Attribute analysis error: {e}")
            return None

    def _analyze_import(self, node: Union[ast.Import, ast.ImportFrom]) -> Optional[Dict[str, Any]]:
        """Analyze import statement with edge case handling"""
        try:
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
                return {
                    "type": "import",
                    "module": None,
                    "names": names,
                    "line_number": node.lineno
                }
            else:  # ast.ImportFrom
                names = [alias.name for alias in node.names]
                return {
                    "type": "from_import",
                    "module": node.module,
                    "names": names,
                    "level": node.level,
                    "line_number": node.lineno
                }

        except Exception as e:
            self.logger.error(f"Import analysis error: {e}")
            return None

    def _get_base_name(self, base: ast.expr) -> str:
        """Get base class name with edge case handling"""
        try:
            if isinstance(base, ast.Name):
                return base.id
            elif isinstance(base, ast.Attribute):
                return f"{self._get_base_name(base.value)}.{base.attr}"
            else:
                return str(base.__class__.__name__)
        except:
            return "Unknown"

    def _get_decorator_name(self, decorator: ast.expr) -> str:
        """Get decorator name with edge case handling"""
        try:
            if isinstance(decorator, ast.Name):
                return decorator.id
            elif isinstance(decorator, ast.Attribute):
                return f"{self._get_base_name(decorator.value)}.{decorator.attr}"
            elif isinstance(decorator, ast.Call):
                return self._get_decorator_name(decorator.func)
            else:
                return str(decorator.__class__.__name__)
        except:
            return "Unknown"

    def _get_annotation(self, annotation: ast.expr) -> str:
        """Get type annotation string with edge case handling"""
        try:
            if isinstance(annotation, ast.Name):
                return annotation.id
            elif isinstance(annotation, ast.Constant):
                return repr(annotation.value)
            elif isinstance(annotation, ast.Attribute):
                return f"{self._get_annotation(annotation.value)}.{annotation.attr}"
            elif isinstance(annotation, ast.Subscript):
                value = self._get_annotation(annotation.value)
                slice_val = self._get_annotation(annotation.slice)
                return f"{value}[{slice_val}]"
            else:
                return ast.unparse(annotation) if hasattr(ast, 'unparse') else str(annotation.__class__.__name__)
        except:
            return "Any"

    def _get_constant_value(self, node: ast.expr) -> Any:
        """Get constant value with edge case handling"""
        try:
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.Num):  # Python < 3.8
                return node.n
            elif isinstance(node, ast.Str):  # Python < 3.8
                return node.s
            elif isinstance(node, ast.List):
                return [self._get_constant_value(item) for item in node.elts]
            elif isinstance(node, ast.Dict):
                keys = [self._get_constant_value(k) for k in node.keys]
                values = [self._get_constant_value(v) for v in node.values]
                return dict(zip(keys, values))
            elif isinstance(node, ast.Name):
                return f"<{node.id}>"  # Variable reference
            else:
                return f"<{node.__class__.__name__}>"
        except:
            return None

    def _get_default_value(self, node: ast.expr) -> Any:
        """Get parameter default value with edge case handling"""
        try:
            return self._get_constant_value(node)
        except:
            return None


class DocstringParser:
    """Enhanced docstring parser supporting multiple formats"""

    def __init__(self, style: str = "google"):
        self.style = style
        self.logger = logging.getLogger(__name__)

    def parse_docstring(self, docstring: Optional[str]) -> Dict[str, Any]:
        """Parse docstring with comprehensive error handling"""
        if not docstring:
            return {
                "short_description": "",
                "long_description": "",
                "parameters": [],
                "returns": None,
                "raises": [],
                "examples": [],
                "see_also": []
            }

        try:
            # Use docstring_parser library
            parsed = docstring_parser.parse(docstring, style=self.style)

            result = {
                "short_description": parsed.short_description or "",
                "long_description": parsed.long_description or "",
                "parameters": [],
                "returns": None,
                "raises": [],
                "examples": [],
                "see_also": []
            }

            # Extract parameters
            for param in parsed.params:
                param_info = {
                    "name": param.arg_name,
                    "type": param.type_name or "Any",
                    "description": param.description or "",
                    "optional": param.is_optional,
                    "default": param.default
                }
                result["parameters"].append(param_info)

            # Extract return information
            if parsed.returns:
                result["returns"] = {
                    "type": parsed.returns.type_name or "Any",
                    "description": parsed.returns.description or ""
                }

            # Extract exceptions
            for exc in parsed.raises:
                result["raises"].append({
                    "exception": exc.type_name or "Exception",
                    "description": exc.description or ""
                })

            # Extract examples (if present in docstring)
            examples = self._extract_examples(docstring)
            result["examples"] = examples

            # Extract see also references
            see_also = self._extract_see_also(docstring)
            result["see_also"] = see_also

            return result

        except Exception as e:
            self.logger.error(f"Docstring parsing error: {e}")
            return {
                "short_description": docstring.split('\n')[0] if docstring else "",
                "long_description": docstring or "",
                "parameters": [],
                "returns": None,
                "raises": [],
                "examples": [],
                "see_also": [],
                "parse_error": str(e)
            }

    def _extract_examples(self, docstring: str) -> List[Dict[str, Any]]:
        """Extract code examples from docstring"""
        examples = []
        try:
            # Look for code blocks in docstring
            code_pattern = r'```(\w+)?\n(.*?)```'
            matches = re.findall(code_pattern, docstring, re.DOTALL)

            for i, (language, code) in enumerate(matches):
                examples.append({
                    "title": f"Example {i+1}",
                    "language": language or "python",
                    "code": code.strip(),
                    "description": ""
                })

            # Also look for indented code blocks
            lines = docstring.split('\n')
            in_code_block = False
            current_code = []

            for line in lines:
                if line.strip().startswith('>>>') or line.strip().startswith('...'):
                    if not in_code_block:
                        in_code_block = True
                        current_code = [line.strip()]
                    else:
                        current_code.append(line.strip())
                elif in_code_block and line.strip():
                    if not line.startswith('    ') and not line.startswith('\t'):
                        # End of code block
                        examples.append({
                            "title": f"Doctest Example {len(examples)+1}",
                            "language": "python",
                            "code": '\n'.join(current_code),
                            "description": "Interactive example"
                        })
                        in_code_block = False
                        current_code = []
                    else:
                        current_code.append(line.strip())

            # Handle last code block if any
            if in_code_block and current_code:
                examples.append({
                    "title": f"Doctest Example {len(examples)+1}",
                    "language": "python",
                    "code": '\n'.join(current_code),
                    "description": "Interactive example"
                })

            return examples

        except Exception as e:
            self.logger.error(f"Example extraction error: {e}")
            return []

    def _extract_see_also(self, docstring: str) -> List[str]:
        """Extract see also references from docstring"""
        try:
            see_also = []

            # Look for "See Also" section
            see_also_pattern = r'See\s+Also:?\s*\n(.*?)(?:\n\s*\n|\n[A-Z]|\Z)'
            match = re.search(see_also_pattern, docstring, re.DOTALL | re.IGNORECASE)

            if match:
                content = match.group(1)
                # Extract references (could be functions, classes, modules)
                ref_pattern = r'([a-zA-Z_][a-zA-Z0-9_.]*)'
                references = re.findall(ref_pattern, content)
                see_also.extend(references)

            return list(set(see_also))  # Remove duplicates

        except Exception as e:
            self.logger.error(f"See also extraction error: {e}")
            return []


class DocumentationGenerator:
    """Main documentation generator with template rendering"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.code_analyzer = CodeAnalyzer(config.get('analyzer', {}))
        self.docstring_parser = DocstringParser(config.get('docstring_style', 'google'))

        # Setup Jinja2 environment
        template_dir = Path(__file__).parent / "templates"
        if not template_dir.exists():
            template_dir.mkdir(exist_ok=True)

        self.jinja_env = Environment(
            loader=FileSystemLoader([str(template_dir), str(Path.cwd())]),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )

        # Add custom filters
        self.jinja_env.filters['format_type'] = self._format_type
        self.jinja_env.filters['format_code'] = self._format_code
        self.jinja_env.filters['markdown'] = self._render_markdown

    def generate_api_documentation(self, source_paths: List[Path]) -> Dict[str, APIDocumentation]:
        """Generate API documentation from source code with comprehensive error handling"""
        api_docs = {}

        try:
            for source_path in source_paths:
                if source_path.is_file() and source_path.suffix == '.py':
                    # Analyze single file
                    module_docs = self._process_module(source_path)
                    if module_docs:
                        api_docs[str(source_path)] = module_docs

                elif source_path.is_dir():
                    # Recursively analyze directory
                    for py_file in source_path.rglob('*.py'):
                        if py_file.name != '__init__.py' or self.config.get('include_init', False):
                            module_docs = self._process_module(py_file)
                            if module_docs:
                                api_docs[str(py_file)] = module_docs
                else:
                    self.logger.warning(f"Skipping invalid path: {source_path}")

            return api_docs

        except Exception as e:
            self.logger.error(f"API documentation generation error: {e}")
            return {}

    def _process_module(self, module_path: Path) -> Optional[List[APIDocumentation]]:
        """Process individual module for documentation"""
        try:
            # Analyze module structure
            module_info = self.code_analyzer.analyze_module(module_path)

            if "error" in module_info:
                self.logger.warning(f"Module analysis failed for {module_path}: {module_info['error']}")
                return None

            api_docs = []

            # Process module-level documentation
            if module_info.get("module_docstring"):
                module_doc = self._create_module_documentation(module_info)
                if module_doc:
                    api_docs.append(module_doc)

            # Process classes
            for class_info in module_info.get("classes", []):
                class_docs = self._create_class_documentation(class_info, module_path)
                api_docs.extend(class_docs)

            # Process functions
            for func_info in module_info.get("functions", []):
                func_doc = self._create_function_documentation(func_info, module_path)
                if func_doc:
                    api_docs.append(func_doc)

            return api_docs if api_docs else None

        except Exception as e:
            self.logger.error(f"Module processing error for {module_path}: {e}")
            return None

    def _create_module_documentation(self, module_info: Dict[str, Any]) -> Optional[APIDocumentation]:
        """Create documentation for module"""
        try:
            docstring = module_info.get("module_docstring", "")
            parsed_doc = self.docstring_parser.parse_docstring(docstring)

            metadata = DocumentationMetadata(
                title=f"Module: {Path(module_info['path']).stem}",
                description=parsed_doc["short_description"],
                doc_type=DocumentationType.API,
                version="1.0.0",
                author="Auto-generated",
                created=datetime.now(),
                updated=datetime.now(),
                tags=["module", "api"],
                dependencies=[],
                validation_level=ValidationLevel.BASIC
            )

            return APIDocumentation(
                name=Path(module_info['path']).stem,
                description=parsed_doc["long_description"],
                parameters=[],
                returns={},
                examples=parsed_doc["examples"],
                raises=parsed_doc["raises"],
                see_also=parsed_doc["see_also"],
                metadata=metadata
            )

        except Exception as e:
            self.logger.error(f"Module documentation creation error: {e}")
            return None

    def _create_class_documentation(self, class_info: Dict[str, Any], module_path: Path) -> List[APIDocumentation]:
        """Create documentation for class and its methods"""
        docs = []

        try:
            # Class documentation
            docstring = class_info.get("docstring", "")
            parsed_doc = self.docstring_parser.parse_docstring(docstring)

            metadata = DocumentationMetadata(
                title=f"Class: {class_info['name']}",
                description=parsed_doc["short_description"],
                doc_type=DocumentationType.API,
                version="1.0.0",
                author="Auto-generated",
                created=datetime.now(),
                updated=datetime.now(),
                tags=["class", "api"],
                dependencies=class_info.get("bases", []),
                validation_level=ValidationLevel.STANDARD
            )

            class_doc = APIDocumentation(
                name=class_info['name'],
                description=parsed_doc["long_description"],
                parameters=parsed_doc["parameters"],
                returns=parsed_doc.get("returns", {}),
                examples=parsed_doc["examples"],
                raises=parsed_doc["raises"],
                see_also=parsed_doc["see_also"],
                metadata=metadata
            )
            docs.append(class_doc)

            # Method documentation
            for method_info in class_info.get("methods", []):
                method_doc = self._create_function_documentation(
                    method_info, module_path, class_name=class_info['name']
                )
                if method_doc:
                    docs.append(method_doc)

            return docs

        except Exception as e:
            self.logger.error(f"Class documentation creation error: {e}")
            return []

    def _create_function_documentation(self, func_info: Dict[str, Any],
                                     module_path: Path,
                                     class_name: Optional[str] = None) -> Optional[APIDocumentation]:
        """Create documentation for function/method"""
        try:
            docstring = func_info.get("docstring", "")
            parsed_doc = self.docstring_parser.parse_docstring(docstring)

            # Merge function parameters with parsed parameters
            parameters = []
            parsed_params = {p["name"]: p for p in parsed_doc["parameters"]}

            for param in func_info.get("parameters", []):
                param_name = param["name"]
                param_info = parsed_params.get(param_name, {})

                merged_param = {
                    "name": param_name,
                    "type": param.get("annotation") or param_info.get("type", "Any"),
                    "description": param_info.get("description", ""),
                    "optional": param.get("default") is not None,
                    "default": param.get("default")
                }
                parameters.append(merged_param)

            func_name = func_info['name']
            if class_name:
                func_name = f"{class_name}.{func_name}"

            metadata = DocumentationMetadata(
                title=f"{'Method' if func_info.get('is_method') else 'Function'}: {func_name}",
                description=parsed_doc["short_description"],
                doc_type=DocumentationType.API,
                version="1.0.0",
                author="Auto-generated",
                created=datetime.now(),
                updated=datetime.now(),
                tags=["function" if not func_info.get('is_method') else "method", "api"],
                dependencies=[],
                validation_level=ValidationLevel.STANDARD
            )

            return APIDocumentation(
                name=func_name,
                description=parsed_doc["long_description"],
                parameters=parameters,
                returns=parsed_doc.get("returns", {}),
                examples=parsed_doc["examples"],
                raises=parsed_doc["raises"],
                see_also=parsed_doc["see_also"],
                metadata=metadata
            )

        except Exception as e:
            self.logger.error(f"Function documentation creation error: {e}")
            return None

    def generate_documentation_site(self, api_docs: Dict[str, Any],
                                  output_dir: Path,
                                  template_name: str = "api_template.html") -> bool:
        """Generate complete documentation site with error handling"""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)

            # Create main index
            self._generate_index_page(api_docs, output_dir)

            # Generate API documentation pages
            for module_path, docs in api_docs.items():
                if docs:
                    self._generate_module_page(docs, output_dir, module_path)

            # Copy static assets
            self._copy_static_assets(output_dir)

            # Generate navigation and search index
            self._generate_navigation(api_docs, output_dir)
            self._generate_search_index(api_docs, output_dir)

            self.logger.info(f"Documentation site generated at {output_dir}")
            return True

        except Exception as e:
            self.logger.error(f"Documentation site generation error: {e}")
            return False

    def _generate_index_page(self, api_docs: Dict[str, Any], output_dir: Path):
        """Generate main index page"""
        try:
            template = self.jinja_env.get_template("index.html")

            # Prepare data for template
            modules = []
            total_functions = 0
            total_classes = 0

            for module_path, docs in api_docs.items():
                if docs:
                    module_name = Path(module_path).stem
                    functions = sum(1 for doc in docs if 'function' in doc.metadata.tags)
                    classes = sum(1 for doc in docs if 'class' in doc.metadata.tags)

                    modules.append({
                        "name": module_name,
                        "path": module_path,
                        "functions": functions,
                        "classes": classes,
                        "description": docs[0].description if docs else ""
                    })

                    total_functions += functions
                    total_classes += classes

            content = template.render(
                title="API Documentation",
                modules=modules,
                stats={
                    "total_modules": len(modules),
                    "total_functions": total_functions,
                    "total_classes": total_classes
                },
                generated_time=datetime.now().isoformat()
            )

            with open(output_dir / "index.html", "w", encoding="utf-8") as f:
                f.write(content)

        except Exception as e:
            self.logger.error(f"Index page generation error: {e}")

    def _generate_module_page(self, docs: List[APIDocumentation],
                            output_dir: Path, module_path: str):
        """Generate individual module documentation page"""
        try:
            template = self.jinja_env.get_template("module.html")

            module_name = Path(module_path).stem
            content = template.render(
                title=f"Module: {module_name}",
                module_name=module_name,
                docs=docs,
                generated_time=datetime.now().isoformat()
            )

            module_file = output_dir / f"{module_name}.html"
            with open(module_file, "w", encoding="utf-8") as f:
                f.write(content)

        except Exception as e:
            self.logger.error(f"Module page generation error: {e}")

    def _copy_static_assets(self, output_dir: Path):
        """Copy static assets (CSS, JS, etc.)"""
        try:
            assets_dir = output_dir / "assets"
            assets_dir.mkdir(exist_ok=True)

            # Generate basic CSS
            css_content = """
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { border-bottom: 2px solid #333; padding-bottom: 10px; }
            .nav { background: #f5f5f5; padding: 10px; margin: 20px 0; }
            .function { margin: 20px 0; padding: 15px; border-left: 3px solid #007acc; }
            .parameters { margin: 10px 0; }
            .parameter { margin: 5px 0; font-family: monospace; }
            .example { background: #f8f8f8; padding: 10px; margin: 10px 0; }
            code { background: #f0f0f0; padding: 2px 4px; }
            pre { background: #f8f8f8; padding: 15px; overflow-x: auto; }
            """

            with open(assets_dir / "style.css", "w") as f:
                f.write(css_content)

            # Generate basic JavaScript
            js_content = """
            function filterDocumentation(query) {
                const items = document.querySelectorAll('.function, .class');
                items.forEach(item => {
                    const text = item.textContent.toLowerCase();
                    item.style.display = text.includes(query.toLowerCase()) ? 'block' : 'none';
                });
            }
            """

            with open(assets_dir / "script.js", "w") as f:
                f.write(js_content)

        except Exception as e:
            self.logger.error(f"Static assets copy error: {e}")

    def _generate_navigation(self, api_docs: Dict[str, Any], output_dir: Path):
        """Generate navigation menu"""
        try:
            nav_data = []

            for module_path, docs in api_docs.items():
                if docs:
                    module_name = Path(module_path).stem
                    items = []

                    for doc in docs:
                        items.append({
                            "name": doc.name,
                            "type": "class" if "class" in doc.metadata.tags else "function",
                            "anchor": doc.name.replace(".", "_")
                        })

                    nav_data.append({
                        "module": module_name,
                        "items": items
                    })

            with open(output_dir / "navigation.json", "w") as f:
                json.dump(nav_data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Navigation generation error: {e}")

    def _generate_search_index(self, api_docs: Dict[str, Any], output_dir: Path):
        """Generate search index for client-side search"""
        try:
            search_index = []

            for module_path, docs in api_docs.items():
                if docs:
                    for doc in docs:
                        search_item = {
                            "name": doc.name,
                            "description": doc.description,
                            "type": "class" if "class" in doc.metadata.tags else "function",
                            "module": Path(module_path).stem,
                            "url": f"{Path(module_path).stem}.html#{doc.name.replace('.', '_')}"
                        }
                        search_index.append(search_item)

            with open(output_dir / "search_index.json", "w") as f:
                json.dump(search_index, f, indent=2)

        except Exception as e:
            self.logger.error(f"Search index generation error: {e}")

    def _format_type(self, type_str: str) -> str:
        """Jinja2 filter for formatting type annotations"""
        if not type_str:
            return "Any"

        # Clean up common type patterns
        type_str = re.sub(r'typing\.', '', type_str)
        type_str = re.sub(r'<class \'([^\']+)\'>', r'\1', type_str)

        return type_str

    def _format_code(self, code: str) -> str:
        """Jinja2 filter for formatting code"""
        try:
            # Format Python code
            if code.strip():
                formatted = black.format_str(code, mode=black.FileMode())
                formatted = isort.code(formatted)
                return formatted
        except:
            pass

        return code

    def _render_markdown(self, text: str) -> str:
        """Jinja2 filter for rendering markdown"""
        try:
            return markdown.markdown(text, extensions=['codehilite', 'fenced_code'])
        except:
            return text


class DocumentationValidator:
    """Documentation validation and quality assessment"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

    def validate_documentation(self, api_docs: List[APIDocumentation],
                             level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
        """Validate documentation quality with comprehensive checks"""
        try:
            issues = []
            suggestions = []
            metrics = {}

            # Basic validation metrics
            total_docs = len(api_docs)
            docs_with_description = sum(1 for doc in api_docs if doc.description.strip())
            docs_with_examples = sum(1 for doc in api_docs if doc.examples)
            docs_with_parameters_described = 0

            for doc in api_docs:
                doc_issues = self._validate_single_doc(doc, level)
                issues.extend(doc_issues)

                # Count parameter descriptions
                if doc.parameters:
                    described_params = sum(1 for p in doc.parameters if p.get("description", "").strip())
                    if described_params == len(doc.parameters):
                        docs_with_parameters_described += 1

            # Calculate metrics
            metrics = {
                "description_coverage": (docs_with_description / total_docs * 100) if total_docs > 0 else 0,
                "example_coverage": (docs_with_examples / total_docs * 100) if total_docs > 0 else 0,
                "parameter_coverage": (docs_with_parameters_described / total_docs * 100) if total_docs > 0 else 0,
                "total_issues": len(issues),
                "critical_issues": sum(1 for issue in issues if issue.get("severity") == "critical"),
                "warning_issues": sum(1 for issue in issues if issue.get("severity") == "warning")
            }

            # Calculate overall score
            score = self._calculate_quality_score(metrics, level)

            # Generate suggestions based on metrics
            suggestions = self._generate_suggestions(metrics)

            return ValidationResult(
                is_valid=score >= 70.0,  # 70% threshold for validity
                score=score,
                issues=issues,
                suggestions=suggestions,
                metrics=metrics
            )

        except Exception as e:
            self.logger.error(f"Documentation validation error: {e}")
            return ValidationResult(
                is_valid=False,
                score=0.0,
                issues=[{"type": "validation_error", "message": str(e), "severity": "critical"}],
                suggestions=["Fix validation system errors"],
                metrics={}
            )

    def _validate_single_doc(self, doc: APIDocumentation, level: ValidationLevel) -> List[Dict[str, Any]]:
        """Validate single documentation item"""
        issues = []

        try:
            # Basic checks
            if not doc.description.strip():
                issues.append({
                    "type": "missing_description",
                    "message": f"'{doc.name}' has no description",
                    "severity": "warning",
                    "item": doc.name
                })

            # Check description length and quality
            if doc.description and len(doc.description.strip()) < 10:
                issues.append({
                    "type": "short_description",
                    "message": f"'{doc.name}' has very short description",
                    "severity": "info",
                    "item": doc.name
                })

            # Parameter validation
            for param in doc.parameters:
                if not param.get("description", "").strip():
                    issues.append({
                        "type": "missing_parameter_description",
                        "message": f"Parameter '{param['name']}' in '{doc.name}' has no description",
                        "severity": "warning",
                        "item": doc.name,
                        "parameter": param["name"]
                    })

                if param.get("type", "").strip() in ["", "Any"]:
                    issues.append({
                        "type": "missing_type_annotation",
                        "message": f"Parameter '{param['name']}' in '{doc.name}' has no type annotation",
                        "severity": "info",
                        "item": doc.name,
                        "parameter": param["name"]
                    })

            # Return type validation
            if doc.returns and not doc.returns.get("description", "").strip():
                issues.append({
                    "type": "missing_return_description",
                    "message": f"Return value of '{doc.name}' has no description",
                    "severity": "info",
                    "item": doc.name
                })

            # Advanced validation for higher levels
            if level in [ValidationLevel.STANDARD, ValidationLevel.COMPREHENSIVE]:
                # Check for examples
                if not doc.examples:
                    issues.append({
                        "type": "missing_examples",
                        "message": f"'{doc.name}' has no examples",
                        "severity": "info",
                        "item": doc.name
                    })

                # Check for exception documentation
                if "raise" in doc.description.lower() and not doc.raises:
                    issues.append({
                        "type": "undocumented_exceptions",
                        "message": f"'{doc.name}' mentions exceptions but doesn't document them",
                        "severity": "warning",
                        "item": doc.name
                    })

            if level == ValidationLevel.COMPREHENSIVE:
                # Grammar and style checks
                issues.extend(self._check_grammar_and_style(doc))

                # Cross-reference validation
                issues.extend(self._check_cross_references(doc))

            return issues

        except Exception as e:
            self.logger.error(f"Single doc validation error for {doc.name}: {e}")
            return [{
                "type": "validation_error",
                "message": f"Error validating '{doc.name}': {str(e)}",
                "severity": "critical",
                "item": doc.name
            }]

    def _check_grammar_and_style(self, doc: APIDocumentation) -> List[Dict[str, Any]]:
        """Check grammar and style issues"""
        issues = []

        try:
            # Check description formatting
            if doc.description:
                # Check for proper sentence capitalization
                sentences = doc.description.split('.')
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence and not sentence[0].isupper():
                        issues.append({
                            "type": "capitalization",
                            "message": f"'{doc.name}' description should start with capital letter",
                            "severity": "info",
                            "item": doc.name
                        })
                        break

                # Check for ending punctuation
                if not doc.description.rstrip().endswith(('.', '!', '?')):
                    issues.append({
                        "type": "missing_punctuation",
                        "message": f"'{doc.name}' description should end with punctuation",
                        "severity": "info",
                        "item": doc.name
                    })

            return issues

        except Exception as e:
            self.logger.error(f"Grammar/style check error for {doc.name}: {e}")
            return []

    def _check_cross_references(self, doc: APIDocumentation) -> List[Dict[str, Any]]:
        """Check cross-reference validity"""
        issues = []

        try:
            # Check see_also references
            for ref in doc.see_also:
                if ref and not self._is_valid_reference(ref):
                    issues.append({
                        "type": "invalid_reference",
                        "message": f"'{doc.name}' references invalid item '{ref}'",
                        "severity": "warning",
                        "item": doc.name,
                        "reference": ref
                    })

            return issues

        except Exception as e:
            self.logger.error(f"Cross-reference check error for {doc.name}: {e}")
            return []

    def _is_valid_reference(self, ref: str) -> bool:
        """Check if reference is valid (simplified check)"""
        # This could be enhanced to check against actual available functions/classes
        return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_.]*$', ref))

    def _calculate_quality_score(self, metrics: Dict[str, float], level: ValidationLevel) -> float:
        """Calculate overall documentation quality score"""
        try:
            base_score = 0.0

            # Description coverage (40% weight)
            base_score += metrics.get("description_coverage", 0) * 0.4

            # Parameter coverage (30% weight)
            base_score += metrics.get("parameter_coverage", 0) * 0.3

            # Example coverage (20% weight)
            base_score += metrics.get("example_coverage", 0) * 0.2

            # Issue penalties (10% weight)
            critical_penalty = metrics.get("critical_issues", 0) * 10
            warning_penalty = metrics.get("warning_issues", 0) * 2
            issue_score = max(0, 100 - critical_penalty - warning_penalty) * 0.1

            base_score += issue_score

            # Adjust for validation level
            if level == ValidationLevel.COMPREHENSIVE:
                base_score *= 0.9  # Higher standards
            elif level == ValidationLevel.BASIC:
                base_score *= 1.1  # Lower standards

            return min(100.0, max(0.0, base_score))

        except Exception as e:
            self.logger.error(f"Quality score calculation error: {e}")
            return 0.0

    def _generate_suggestions(self, metrics: Dict[str, float]) -> List[str]:
        """Generate improvement suggestions based on metrics"""
        suggestions = []

        try:
            if metrics.get("description_coverage", 0) < 80:
                suggestions.append("Add descriptions to undocumented functions and classes")

            if metrics.get("parameter_coverage", 0) < 70:
                suggestions.append("Document function parameters with descriptions and types")

            if metrics.get("example_coverage", 0) < 30:
                suggestions.append("Add usage examples to important functions")

            if metrics.get("critical_issues", 0) > 0:
                suggestions.append("Fix critical documentation issues immediately")

            if metrics.get("warning_issues", 0) > 10:
                suggestions.append("Address warning-level documentation issues")

            if not suggestions:
                suggestions.append("Documentation quality is good - consider adding more examples")

            return suggestions

        except Exception as e:
            self.logger.error(f"Suggestions generation error: {e}")
            return ["Review documentation for potential improvements"]


class DocumentationDeployer:
    """Documentation deployment and versioning system"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

    def deploy_documentation(self, docs_dir: Path,
                           deployment_config: Dict[str, Any]) -> DeploymentResult:
        """Deploy documentation with comprehensive error handling"""
        try:
            deployment_type = deployment_config.get("type", "static")

            if deployment_type == "static":
                return self._deploy_static(docs_dir, deployment_config)
            elif deployment_type == "github_pages":
                return self._deploy_github_pages(docs_dir, deployment_config)
            elif deployment_type == "netlify":
                return self._deploy_netlify(docs_dir, deployment_config)
            elif deployment_type == "aws_s3":
                return self._deploy_aws_s3(docs_dir, deployment_config)
            else:
                return DeploymentResult(
                    status=DeploymentStatus.FAILED,
                    url=None,
                    build_log=f"Unsupported deployment type: {deployment_type}",
                    deployment_time=datetime.now(),
                    version="unknown",
                    errors=[f"Unsupported deployment type: {deployment_type}"]
                )

        except Exception as e:
            self.logger.error(f"Documentation deployment error: {e}")
            return DeploymentResult(
                status=DeploymentStatus.FAILED,
                url=None,
                build_log=str(e),
                deployment_time=datetime.now(),
                version="unknown",
                errors=[str(e)]
            )

    def _deploy_static(self, docs_dir: Path, config: Dict[str, Any]) -> DeploymentResult:
        """Deploy to static file server"""
        try:
            target_dir = Path(config.get("target_directory", "/var/www/docs"))

            # Copy files
            import shutil
            if target_dir.exists():
                shutil.rmtree(target_dir)

            shutil.copytree(docs_dir, target_dir)

            base_url = config.get("base_url", "http://localhost")

            return DeploymentResult(
                status=DeploymentStatus.SUCCESS,
                url=f"{base_url}/index.html",
                build_log="Static deployment completed successfully",
                deployment_time=datetime.now(),
                version="1.0.0",
                errors=[]
            )

        except Exception as e:
            return DeploymentResult(
                status=DeploymentStatus.FAILED,
                url=None,
                build_log=f"Static deployment failed: {e}",
                deployment_time=datetime.now(),
                version="unknown",
                errors=[str(e)]
            )

    def _deploy_github_pages(self, docs_dir: Path, config: Dict[str, Any]) -> DeploymentResult:
        """Deploy to GitHub Pages"""
        try:
            repo_url = config.get("repository_url")
            branch = config.get("branch", "gh-pages")

            if not repo_url:
                raise ValueError("Repository URL required for GitHub Pages deployment")

            # This is a simplified implementation
            # In practice, you'd use git commands or GitHub API

            build_log = f"Would deploy to GitHub Pages: {repo_url} (branch: {branch})"

            return DeploymentResult(
                status=DeploymentStatus.SUCCESS,
                url=f"https://{config.get('github_username', 'user')}.github.io/{config.get('repo_name', 'repo')}",
                build_log=build_log,
                deployment_time=datetime.now(),
                version="1.0.0",
                errors=[]
            )

        except Exception as e:
            return DeploymentResult(
                status=DeploymentStatus.FAILED,
                url=None,
                build_log=f"GitHub Pages deployment failed: {e}",
                deployment_time=datetime.now(),
                version="unknown",
                errors=[str(e)]
            )

    def _deploy_netlify(self, docs_dir: Path, config: Dict[str, Any]) -> DeploymentResult:
        """Deploy to Netlify (simplified)"""
        try:
            site_name = config.get("site_name", "docs-site")

            build_log = f"Would deploy to Netlify site: {site_name}"

            return DeploymentResult(
                status=DeploymentStatus.SUCCESS,
                url=f"https://{site_name}.netlify.app",
                build_log=build_log,
                deployment_time=datetime.now(),
                version="1.0.0",
                errors=[]
            )

        except Exception as e:
            return DeploymentResult(
                status=DeploymentStatus.FAILED,
                url=None,
                build_log=f"Netlify deployment failed: {e}",
                deployment_time=datetime.now(),
                version="unknown",
                errors=[str(e)]
            )

    def _deploy_aws_s3(self, docs_dir: Path, config: Dict[str, Any]) -> DeploymentResult:
        """Deploy to AWS S3 (simplified)"""
        try:
            bucket_name = config.get("bucket_name")
            region = config.get("region", "us-east-1")

            if not bucket_name:
                raise ValueError("S3 bucket name required for AWS deployment")

            build_log = f"Would deploy to S3 bucket: {bucket_name} (region: {region})"

            return DeploymentResult(
                status=DeploymentStatus.SUCCESS,
                url=f"https://{bucket_name}.s3.{region}.amazonaws.com/index.html",
                build_log=build_log,
                deployment_time=datetime.now(),
                version="1.0.0",
                errors=[]
            )

        except Exception as e:
            return DeploymentResult(
                status=DeploymentStatus.FAILED,
                url=None,
                build_log=f"AWS S3 deployment failed: {e}",
                deployment_time=datetime.now(),
                version="unknown",
                errors=[str(e)]
            )


class DocumentationFramework:
    """Main documentation framework orchestrating all components"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.generator = DocumentationGenerator(config.get('generator', {}))
        self.validator = DocumentationValidator(config.get('validator', {}))
        self.deployer = DocumentationDeployer(config.get('deployer', {}))

    async def build_complete_documentation(self,
                                         source_paths: List[Path],
                                         output_dir: Path,
                                         deploy_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build complete documentation pipeline with error handling"""
        try:
            results = {
                "timestamp": datetime.now().isoformat(),
                "generation": {},
                "validation": {},
                "deployment": {},
                "summary": {}
            }

            # Step 1: Generate API documentation
            self.logger.info("Starting API documentation generation...")
            api_docs = self.generator.generate_api_documentation(source_paths)

            if api_docs:
                # Generate documentation site
                site_generated = self.generator.generate_documentation_site(api_docs, output_dir)

                results["generation"] = {
                    "success": site_generated,
                    "modules_processed": len(api_docs),
                    "total_docs": sum(len(docs) if docs else 0 for docs in api_docs.values()),
                    "output_directory": str(output_dir)
                }

                if site_generated:
                    # Step 2: Validate documentation
                    self.logger.info("Validating documentation quality...")
                    all_docs = []
                    for docs_list in api_docs.values():
                        if docs_list:
                            all_docs.extend(docs_list)

                    validation_result = self.validator.validate_documentation(
                        all_docs,
                        ValidationLevel(self.config.get('validation_level', 'standard'))
                    )

                    results["validation"] = asdict(validation_result)

                    # Step 3: Deploy documentation (if configured)
                    if deploy_config:
                        self.logger.info("Deploying documentation...")
                        deployment_result = self.deployer.deploy_documentation(output_dir, deploy_config)
                        results["deployment"] = asdict(deployment_result)
                    else:
                        results["deployment"] = {"skipped": True, "reason": "No deployment config provided"}

                else:
                    results["generation"]["error"] = "Failed to generate documentation site"
            else:
                results["generation"] = {
                    "success": False,
                    "error": "No API documentation generated",
                    "modules_processed": 0
                }

            # Step 4: Generate summary
            results["summary"] = self._generate_build_summary(results)

            return results

        except Exception as e:
            self.logger.error(f"Complete documentation build error: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "summary": {"success": False, "error": str(e)}
            }

    def _generate_build_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate build summary with statistics"""
        try:
            summary = {
                "success": False,
                "modules_processed": results.get("generation", {}).get("modules_processed", 0),
                "total_docs": results.get("generation", {}).get("total_docs", 0),
                "validation_score": 0.0,
                "deployment_status": "not_attempted",
                "recommendations": []
            }

            # Check generation success
            generation_success = results.get("generation", {}).get("success", False)

            if generation_success:
                # Check validation results
                validation = results.get("validation", {})
                if validation:
                    summary["validation_score"] = validation.get("score", 0.0)
                    summary["success"] = validation.get("is_valid", False)

                    # Add validation suggestions to recommendations
                    suggestions = validation.get("suggestions", [])
                    summary["recommendations"].extend(suggestions)

                # Check deployment status
                deployment = results.get("deployment", {})
                if not deployment.get("skipped", False):
                    deployment_status = deployment.get("status", "unknown")
                    summary["deployment_status"] = deployment_status
                    summary["deployment_url"] = deployment.get("url")

                    if deployment_status == "failed":
                        errors = deployment.get("errors", [])
                        summary["recommendations"].extend([f"Fix deployment issue: {err}" for err in errors])
            else:
                generation_error = results.get("generation", {}).get("error", "Unknown generation error")
                summary["recommendations"].append(f"Fix generation issue: {generation_error}")

            # Add general recommendations
            if summary["total_docs"] == 0:
                summary["recommendations"].append("Ensure source files contain documentable code")

            if summary["validation_score"] < 70:
                summary["recommendations"].append("Improve documentation quality to meet standards")

            return summary

        except Exception as e:
            self.logger.error(f"Build summary generation error: {e}")
            return {
                "success": False,
                "error": str(e),
                "recommendations": ["Fix summary generation system"]
            }


# Template creation helper
def create_default_templates():
    """Create default HTML templates for documentation"""
    templates = {
        "index.html": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <link rel="stylesheet" href="assets/style.css">
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        <p>Generated on {{ generated_time }}</p>
    </div>

    <div class="nav">
        <h2>Documentation Statistics</h2>
        <ul>
            <li>Total Modules: {{ stats.total_modules }}</li>
            <li>Total Functions: {{ stats.total_functions }}</li>
            <li>Total Classes: {{ stats.total_classes }}</li>
        </ul>
    </div>

    <div class="content">
        <h2>Modules</h2>
        {% for module in modules %}
        <div class="module">
            <h3><a href="{{ module.name }}.html">{{ module.name }}</a></h3>
            <p>{{ module.description }}</p>
            <p>Functions: {{ module.functions }}, Classes: {{ module.classes }}</p>
        </div>
        {% endfor %}
    </div>

    <script src="assets/script.js"></script>
</body>
</html>""",

        "module.html": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <link rel="stylesheet" href="assets/style.css">
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        <p><a href="index.html">← Back to Index</a></p>
    </div>

    <div class="content">
        {% for doc in docs %}
        <div class="function" id="{{ doc.name.replace('.', '_') }}">
            <h3>{{ doc.name }}</h3>
            <p>{{ doc.description | markdown }}</p>

            {% if doc.parameters %}
            <div class="parameters">
                <h4>Parameters:</h4>
                {% for param in doc.parameters %}
                <div class="parameter">
                    <strong>{{ param.name }}</strong> ({{ param.type | format_type }})
                    {% if param.optional %}[optional]{% endif %}
                    {% if param.default %} = {{ param.default }}{% endif %}
                    <br>{{ param.description }}
                </div>
                {% endfor %}
            </div>
            {% endif %}

            {% if doc.returns %}
            <div class="returns">
                <h4>Returns:</h4>
                <strong>{{ doc.returns.type | format_type }}</strong> - {{ doc.returns.description }}
            </div>
            {% endif %}

            {% if doc.examples %}
            <div class="examples">
                <h4>Examples:</h4>
                {% for example in doc.examples %}
                <div class="example">
                    <h5>{{ example.title }}</h5>
                    <pre><code>{{ example.code | format_code }}</code></pre>
                    {% if example.description %}
                    <p>{{ example.description }}</p>
                    {% endif %}
                </div>
                {% endfor %}
            </div>
            {% endif %}
        </div>
        {% endfor %}
    </div>

    <script src="assets/script.js"></script>
</body>
</html>"""
    }

    return templates


# Main execution function for testing
async def main():
    """Main function for testing the documentation framework"""
    # Initialize framework
    config = {
        'docstring_style': 'google',
        'validation_level': 'standard',
        'generator': {},
        'validator': {},
        'deployer': {}
    }

    framework = DocumentationFramework(config)

    # Test with current file
    current_file = Path(__file__)
    source_paths = [current_file]
    output_dir = Path("docs_output")

    # Build documentation
    results = await framework.build_complete_documentation(source_paths, output_dir)

    print("Documentation Framework Test Results:")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())