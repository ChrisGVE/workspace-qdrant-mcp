"""
Project Data Generator

Generates dummy project context data including Git repositories, language
configurations, LSP server data, and workspace structures for testing
project detection and multi-tenant scenarios.
"""

import random
import uuid
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import json


@dataclass
class ProjectSpec:
    """Specification for project generation."""
    name: str
    languages: List[str]
    has_git: bool = True
    has_submodules: bool = False
    complexity: str = "medium"  # simple, medium, complex


@dataclass
class LanguageConfig:
    """Language configuration specification."""
    name: str
    extensions: List[str]
    lsp_server: Optional[str] = None
    tree_sitter_grammar: Optional[str] = None
    package_managers: List[str] = None


class ProjectDataGenerator:
    """Generates realistic project and workspace data."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional random seed for reproducibility."""
        self.seed = seed or int(time.time())
        random.seed(self.seed)

        self._init_language_configs()
        self._init_project_templates()

    def _init_language_configs(self):
        """Initialize language configurations based on internal_configuration.yaml."""
        self.language_configs = {
            # Popular programming languages
            "python": LanguageConfig(
                name="python",
                extensions=[".py", ".pyx", ".pyi", ".pyw"],
                lsp_server="pylsp",
                tree_sitter_grammar="python",
                package_managers=["pip", "poetry", "pipenv", "conda"]
            ),
            "rust": LanguageConfig(
                name="rust",
                extensions=[".rs"],
                lsp_server="rust-analyzer",
                tree_sitter_grammar="rust",
                package_managers=["cargo"]
            ),
            "javascript": LanguageConfig(
                name="javascript",
                extensions=[".js", ".jsx", ".mjs", ".cjs"],
                lsp_server="typescript-language-server",
                tree_sitter_grammar="javascript",
                package_managers=["npm", "yarn", "pnpm"]
            ),
            "typescript": LanguageConfig(
                name="typescript",
                extensions=[".ts", ".tsx"],
                lsp_server="typescript-language-server",
                tree_sitter_grammar="typescript",
                package_managers=["npm", "yarn", "pnpm"]
            ),
            "java": LanguageConfig(
                name="java",
                extensions=[".java"],
                lsp_server="jdtls",
                tree_sitter_grammar="java",
                package_managers=["maven", "gradle"]
            ),
            "cpp": LanguageConfig(
                name="cpp",
                extensions=[".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hh", ".hxx"],
                lsp_server="clangd",
                tree_sitter_grammar="cpp",
                package_managers=["conan", "vcpkg"]
            ),
            "c": LanguageConfig(
                name="c",
                extensions=[".c", ".h"],
                lsp_server="clangd",
                tree_sitter_grammar="c",
                package_managers=["conan", "vcpkg"]
            ),
            "go": LanguageConfig(
                name="go",
                extensions=[".go"],
                lsp_server="gopls",
                tree_sitter_grammar="go",
                package_managers=["go mod"]
            ),
            "kotlin": LanguageConfig(
                name="kotlin",
                extensions=[".kt", ".kts"],
                lsp_server="kotlin-language-server",
                tree_sitter_grammar="kotlin",
                package_managers=["gradle", "maven"]
            ),
            "swift": LanguageConfig(
                name="swift",
                extensions=[".swift"],
                lsp_server="sourcekit-lsp",
                tree_sitter_grammar="swift",
                package_managers=["swift package manager"]
            ),
            "csharp": LanguageConfig(
                name="csharp",
                extensions=[".cs"],
                lsp_server="omnisharp",
                tree_sitter_grammar="c_sharp",
                package_managers=["nuget", "dotnet"]
            ),
            "ruby": LanguageConfig(
                name="ruby",
                extensions=[".rb", ".rake", ".gemspec"],
                lsp_server="solargraph",
                tree_sitter_grammar="ruby",
                package_managers=["gem", "bundler"]
            ),
            "php": LanguageConfig(
                name="php",
                extensions=[".php", ".phtml"],
                lsp_server="phpactor",
                tree_sitter_grammar="php",
                package_managers=["composer"]
            ),
            "scala": LanguageConfig(
                name="scala",
                extensions=[".scala", ".sc"],
                lsp_server="metals",
                tree_sitter_grammar="scala",
                package_managers=["sbt", "mill"]
            ),
            "haskell": LanguageConfig(
                name="haskell",
                extensions=[".hs", ".lhs"],
                lsp_server="haskell-language-server",
                tree_sitter_grammar="haskell",
                package_managers=["cabal", "stack"]
            ),
            "clojure": LanguageConfig(
                name="clojure",
                extensions=[".clj", ".cljs", ".cljc"],
                lsp_server="clojure-lsp",
                tree_sitter_grammar="clojure",
                package_managers=["leiningen", "boot", "deps.edn"]
            ),
            "erlang": LanguageConfig(
                name="erlang",
                extensions=[".erl", ".hrl"],
                lsp_server="erlang_ls",
                tree_sitter_grammar="erlang",
                package_managers=["rebar3"]
            ),
            "elixir": LanguageConfig(
                name="elixir",
                extensions=[".ex", ".exs"],
                lsp_server="elixir-ls",
                tree_sitter_grammar="elixir",
                package_managers=["mix"]
            ),
            "dart": LanguageConfig(
                name="dart",
                extensions=[".dart"],
                lsp_server="dart",
                tree_sitter_grammar="dart",
                package_managers=["pub"]
            ),
            "lua": LanguageConfig(
                name="lua",
                extensions=[".lua"],
                lsp_server="lua-language-server",
                tree_sitter_grammar="lua",
                package_managers=["luarocks"]
            ),
            # Markup and data formats
            "markdown": LanguageConfig(
                name="markdown",
                extensions=[".md", ".markdown", ".mdown"],
                lsp_server=None,
                tree_sitter_grammar="markdown",
                package_managers=[]
            ),
            "json": LanguageConfig(
                name="json",
                extensions=[".json", ".jsonc"],
                lsp_server="json-languageserver",
                tree_sitter_grammar="json",
                package_managers=[]
            ),
            "yaml": LanguageConfig(
                name="yaml",
                extensions=[".yaml", ".yml"],
                lsp_server="yaml-language-server",
                tree_sitter_grammar="yaml",
                package_managers=[]
            ),
            "toml": LanguageConfig(
                name="toml",
                extensions=[".toml"],
                lsp_server="taplo",
                tree_sitter_grammar="toml",
                package_managers=[]
            ),
            "xml": LanguageConfig(
                name="xml",
                extensions=[".xml", ".xsd", ".xsl"],
                lsp_server="lemminx",
                tree_sitter_grammar="xml",
                package_managers=[]
            ),
            "html": LanguageConfig(
                name="html",
                extensions=[".html", ".htm"],
                lsp_server="vscode-html-language-server",
                tree_sitter_grammar="html",
                package_managers=[]
            ),
            "css": LanguageConfig(
                name="css",
                extensions=[".css"],
                lsp_server="vscode-css-language-server",
                tree_sitter_grammar="css",
                package_managers=[]
            ),
            # Shell and config languages
            "bash": LanguageConfig(
                name="bash",
                extensions=[".sh", ".bash", ".zsh"],
                lsp_server="bash-language-server",
                tree_sitter_grammar="bash",
                package_managers=[]
            ),
            "dockerfile": LanguageConfig(
                name="dockerfile",
                extensions=["Dockerfile", ".dockerfile"],
                lsp_server="docker-langserver",
                tree_sitter_grammar="dockerfile",
                package_managers=[]
            ),
            # Specialized languages
            "sql": LanguageConfig(
                name="sql",
                extensions=[".sql"],
                lsp_server="sqls",
                tree_sitter_grammar="sql",
                package_managers=[]
            ),
            "graphql": LanguageConfig(
                name="graphql",
                extensions=[".graphql", ".gql"],
                lsp_server="graphql-language-service",
                tree_sitter_grammar="graphql",
                package_managers=[]
            )
        }

    def _init_project_templates(self):
        """Initialize project templates for different types."""
        self.project_templates = {
            "workspace-qdrant-mcp": {
                "type": "vector_database_mcp",
                "languages": ["python", "rust", "yaml", "toml", "markdown"],
                "structure": {
                    "src/": ["python", "rust"],
                    "tests/": ["python", "rust"],
                    "docs/": ["markdown"],
                    "config/": ["yaml", "toml"],
                    "scripts/": ["bash", "python"]
                },
                "package_files": ["pyproject.toml", "Cargo.toml", "requirements.txt"],
                "has_submodules": False,
                "complexity": "complex"
            },
            "web-application": {
                "type": "fullstack_web",
                "languages": ["typescript", "javascript", "html", "css", "json"],
                "structure": {
                    "frontend/": ["typescript", "html", "css"],
                    "backend/": ["typescript", "javascript"],
                    "shared/": ["typescript"],
                    "config/": ["json", "yaml"],
                    "docs/": ["markdown"]
                },
                "package_files": ["package.json", "tsconfig.json"],
                "has_submodules": True,
                "complexity": "medium"
            },
            "microservice": {
                "type": "microservice",
                "languages": ["java", "yaml", "dockerfile", "sql"],
                "structure": {
                    "src/main/java/": ["java"],
                    "src/test/java/": ["java"],
                    "src/main/resources/": ["yaml", "sql"],
                    "docker/": ["dockerfile"],
                    "docs/": ["markdown"]
                },
                "package_files": ["pom.xml", "application.yml"],
                "has_submodules": False,
                "complexity": "medium"
            },
            "ml-pipeline": {
                "type": "machine_learning",
                "languages": ["python", "yaml", "json", "markdown"],
                "structure": {
                    "src/": ["python"],
                    "notebooks/": ["python"],
                    "data/": [],
                    "models/": [],
                    "config/": ["yaml", "json"],
                    "tests/": ["python"]
                },
                "package_files": ["requirements.txt", "pyproject.toml", "environment.yml"],
                "has_submodules": False,
                "complexity": "medium"
            },
            "cli-tool": {
                "type": "command_line_tool",
                "languages": ["go", "yaml", "markdown"],
                "structure": {
                    "cmd/": ["go"],
                    "internal/": ["go"],
                    "pkg/": ["go"],
                    "config/": ["yaml"],
                    "docs/": ["markdown"]
                },
                "package_files": ["go.mod", "go.sum"],
                "has_submodules": False,
                "complexity": "simple"
            },
            "mobile-app": {
                "type": "mobile_application",
                "languages": ["dart", "yaml", "json"],
                "structure": {
                    "lib/": ["dart"],
                    "test/": ["dart"],
                    "android/": ["java", "kotlin"],
                    "ios/": ["swift"],
                    "assets/": [],
                    "config/": ["yaml"]
                },
                "package_files": ["pubspec.yaml"],
                "has_submodules": False,
                "complexity": "complex"
            }
        }

    def generate_project_context(self, project_spec: Optional[ProjectSpec] = None) -> Dict[str, Any]:
        """Generate comprehensive project context data."""
        if project_spec is None:
            # Generate random project
            template_name = random.choice(list(self.project_templates.keys()))
            template = self.project_templates[template_name]
            project_spec = ProjectSpec(
                name=f"{template_name}_{random.randint(100, 999)}",
                languages=template["languages"],
                has_git=random.choice([True, False]),
                has_submodules=template["has_submodules"],
                complexity=template["complexity"]
            )

        project_data = {
            "project_name": project_spec.name,
            "project_type": self._get_project_type(project_spec),
            "languages": self._generate_language_data(project_spec.languages),
            "structure": self._generate_directory_structure(project_spec),
            "git": self._generate_git_context(project_spec) if project_spec.has_git else None,
            "workspace": self._generate_workspace_config(project_spec),
            "metadata": self._generate_project_metadata(project_spec),
            "lsp_servers": self._generate_lsp_server_data(project_spec.languages),
            "collections": self._generate_collection_mapping(project_spec)
        }

        return project_data

    def _get_project_type(self, project_spec: ProjectSpec) -> str:
        """Determine project type based on languages and structure."""
        languages = set(project_spec.languages)

        if "rust" in languages and "python" in languages:
            return "mixed_systems_programming"
        elif "typescript" in languages or "javascript" in languages:
            return "web_application"
        elif "java" in languages or "kotlin" in languages:
            return "jvm_application"
        elif "python" in languages:
            return "python_application"
        elif "rust" in languages:
            return "systems_programming"
        elif "go" in languages:
            return "go_application"
        else:
            return "multi_language_project"

    def _generate_language_data(self, languages: List[str]) -> Dict[str, Dict[str, Any]]:
        """Generate detailed language configuration data."""
        language_data = {}

        for lang in languages:
            if lang in self.language_configs:
                config = self.language_configs[lang]
                language_data[lang] = {
                    "name": config.name,
                    "extensions": config.extensions,
                    "lsp_server": config.lsp_server,
                    "tree_sitter_grammar": config.tree_sitter_grammar,
                    "package_managers": config.package_managers or [],
                    "file_count": random.randint(1, 500),
                    "lines_of_code": random.randint(100, 50000),
                    "last_modified": int(time.time() - random.randint(0, 86400 * 30)),
                    "complexity_score": random.uniform(0.1, 1.0)
                }
            else:
                # Unknown language
                language_data[lang] = {
                    "name": lang,
                    "extensions": [f".{lang}"],
                    "lsp_server": None,
                    "tree_sitter_grammar": None,
                    "package_managers": [],
                    "file_count": random.randint(1, 10),
                    "lines_of_code": random.randint(10, 1000),
                    "last_modified": int(time.time() - random.randint(0, 86400)),
                    "complexity_score": 0.1
                }

        return language_data

    def _generate_directory_structure(self, project_spec: ProjectSpec) -> Dict[str, Any]:
        """Generate realistic directory structure."""
        # Start with common directories
        structure = {
            "root": {
                "type": "directory",
                "children": {},
                "file_count": 0,
                "total_size_bytes": 0
            }
        }

        # Get template if available
        template_name = None
        for name, template in self.project_templates.items():
            if set(template["languages"]).intersection(set(project_spec.languages)):
                template_name = name
                break

        if template_name and template_name in self.project_templates:
            template = self.project_templates[template_name]
            for directory, langs in template["structure"].items():
                self._add_directory_to_structure(structure["root"], directory, langs)
        else:
            # Generate generic structure
            common_dirs = ["src", "tests", "docs", "config", "scripts"]
            for directory in common_dirs:
                if random.random() > 0.3:  # 70% chance to include each directory
                    self._add_directory_to_structure(
                        structure["root"],
                        f"{directory}/",
                        random.sample(project_spec.languages, random.randint(1, len(project_spec.languages)))
                    )

        # Calculate totals
        self._calculate_structure_totals(structure["root"])

        return structure

    def _add_directory_to_structure(self, parent: Dict[str, Any], path: str, languages: List[str]):
        """Add directory and files to structure."""
        parts = path.strip("/").split("/")
        current = parent

        # Navigate/create directory structure
        for part in parts[:-1] if len(parts) > 1 else [parts[0]]:
            if part not in current["children"]:
                current["children"][part] = {
                    "type": "directory",
                    "children": {},
                    "file_count": 0,
                    "total_size_bytes": 0
                }
            current = current["children"][part]

        # Add files for the specified languages
        for lang in languages:
            if lang in self.language_configs:
                config = self.language_configs[lang]
                file_count = random.randint(1, 20)

                for i in range(file_count):
                    extension = random.choice(config.extensions)
                    filename = f"file_{i}{extension}"
                    size = random.randint(100, 10000)

                    current["children"][filename] = {
                        "type": "file",
                        "language": lang,
                        "size_bytes": size,
                        "lines": random.randint(10, 500),
                        "last_modified": int(time.time() - random.randint(0, 86400 * 7))
                    }

    def _calculate_structure_totals(self, node: Dict[str, Any]):
        """Recursively calculate file counts and sizes."""
        if node["type"] == "file":
            return 1, node["size_bytes"]

        total_files = 0
        total_size = 0

        for child in node["children"].values():
            files, size = self._calculate_structure_totals(child)
            total_files += files
            total_size += size

        node["file_count"] = total_files
        node["total_size_bytes"] = total_size

        return total_files, total_size

    def _generate_git_context(self, project_spec: ProjectSpec) -> Dict[str, Any]:
        """Generate Git repository context."""
        branches = ["main", "develop", "feature/test", "hotfix/urgent", "release/v1.0"]
        current_branch = random.choice(branches)

        git_data = {
            "repository": {
                "url": f"https://github.com/user/{project_spec.name}.git",
                "provider": "github",
                "private": random.choice([True, False]),
                "default_branch": "main",
                "current_branch": current_branch,
                "is_dirty": random.choice([True, False]),
                "ahead": random.randint(0, 10),
                "behind": random.randint(0, 5)
            },
            "commit": {
                "hash": uuid.uuid4().hex[:8],
                "short_hash": uuid.uuid4().hex[:7],
                "message": random.choice([
                    "feat: add new feature",
                    "fix: resolve critical bug",
                    "docs: update documentation",
                    "refactor: improve code structure",
                    "test: add comprehensive tests"
                ]),
                "author": random.choice(["Alice Smith", "Bob Jones", "Charlie Brown"]),
                "email": f"user{random.randint(1, 100)}@example.com",
                "timestamp": int(time.time() - random.randint(0, 86400 * 7))
            },
            "branches": {
                "local": random.sample(branches, random.randint(2, len(branches))),
                "remote": random.sample(branches, random.randint(1, len(branches)))
            },
            "remotes": {
                "origin": f"https://github.com/user/{project_spec.name}.git",
                "upstream": f"https://github.com/upstream/{project_spec.name}.git" if random.random() > 0.7 else None
            },
            "status": {
                "staged": random.randint(0, 5),
                "modified": random.randint(0, 10),
                "untracked": random.randint(0, 8),
                "deleted": random.randint(0, 2)
            }
        }

        if project_spec.has_submodules:
            git_data["submodules"] = self._generate_submodules()

        return git_data

    def _generate_submodules(self) -> List[Dict[str, Any]]:
        """Generate Git submodule data."""
        submodules = []

        for i in range(random.randint(1, 3)):
            submodules.append({
                "name": f"submodule_{i}",
                "path": f"external/submodule_{i}",
                "url": f"https://github.com/external/submodule_{i}.git",
                "branch": random.choice(["main", "master", "develop"]),
                "commit": uuid.uuid4().hex[:8],
                "status": random.choice(["up-to-date", "ahead", "behind", "modified"])
            })

        return submodules

    def _generate_workspace_config(self, project_spec: ProjectSpec) -> Dict[str, Any]:
        """Generate workspace configuration data."""
        workspace_types = ["docs", "notes", "scratchbook", "knowledge", "context", "memory"]

        workspace_config = {
            "project_root": f"/tmp/projects/{project_spec.name}",
            "workspace_types": random.sample(workspace_types, random.randint(3, len(workspace_types))),
            "auto_watch": random.choice([True, False]),
            "include_patterns": ["*.py", "*.rs", "*.js", "*.md", "*.json", "*.yaml"],
            "exclude_patterns": [
                "node_modules/", ".git/", "__pycache__/", "target/",
                "*.log", "*.tmp", ".env", ".venv/"
            ],
            "max_file_size_mb": random.randint(1, 100),
            "processing_options": {
                "extract_text": True,
                "generate_embeddings": True,
                "detect_language": True,
                "chunk_size": random.choice([512, 1024, 2048]),
                "overlap_size": random.choice([50, 100, 200])
            },
            "collection_settings": {
                "vector_size": random.choice([384, 768, 1536]),
                "distance_metric": random.choice(["Cosine", "Euclidean", "Dot"]),
                "auto_optimize": True,
                "replication_factor": random.randint(1, 3)
            }
        }

        return workspace_config

    def _generate_project_metadata(self, project_spec: ProjectSpec) -> Dict[str, Any]:
        """Generate project metadata."""
        return {
            "created_at": int(time.time() - random.randint(86400, 86400 * 365)),
            "last_accessed": int(time.time() - random.randint(0, 86400 * 7)),
            "complexity": project_spec.complexity,
            "estimated_size": random.choice(["small", "medium", "large", "enterprise"]),
            "team_size": random.randint(1, 20),
            "development_stage": random.choice(["prototype", "development", "testing", "production", "maintenance"]),
            "license": random.choice(["MIT", "Apache-2.0", "GPL-3.0", "BSD-3-Clause", "proprietary"]),
            "documentation_coverage": random.uniform(0.0, 1.0),
            "test_coverage": random.uniform(0.0, 1.0),
            "dependencies": {
                "direct": random.randint(5, 50),
                "total": random.randint(20, 500),
                "outdated": random.randint(0, 10),
                "security_issues": random.randint(0, 3)
            },
            "activity": {
                "commits_last_month": random.randint(0, 100),
                "contributors": random.randint(1, 10),
                "issues_open": random.randint(0, 50),
                "pull_requests_open": random.randint(0, 10)
            }
        }

    def _generate_lsp_server_data(self, languages: List[str]) -> Dict[str, Any]:
        """Generate LSP server configuration and status data."""
        lsp_data = {}

        for lang in languages:
            if lang in self.language_configs:
                config = self.language_configs[lang]
                if config.lsp_server:
                    lsp_data[lang] = {
                        "server_name": config.lsp_server,
                        "status": random.choice(["running", "stopped", "error", "starting"]),
                        "version": f"v{random.randint(1, 5)}.{random.randint(0, 20)}.{random.randint(0, 50)}",
                        "capabilities": self._generate_lsp_capabilities(),
                        "performance": {
                            "avg_response_time_ms": random.randint(10, 1000),
                            "memory_usage_mb": random.randint(10, 500),
                            "cpu_usage_percent": random.uniform(0.0, 50.0),
                            "active_documents": random.randint(0, 100)
                        },
                        "configuration": {
                            "workspace_folders": [f"/tmp/projects/{random.randint(1, 10)}"],
                            "initialization_options": {},
                            "settings": self._generate_lsp_settings(config.lsp_server)
                        },
                        "diagnostics": {
                            "errors": random.randint(0, 20),
                            "warnings": random.randint(0, 50),
                            "hints": random.randint(0, 100),
                            "last_run": int(time.time() - random.randint(0, 3600))
                        }
                    }

        return lsp_data

    def _generate_lsp_capabilities(self) -> Dict[str, Any]:
        """Generate LSP server capabilities."""
        return {
            "textDocumentSync": random.choice([1, 2]),  # None, Full, Incremental
            "completionProvider": {
                "resolveProvider": random.choice([True, False]),
                "triggerCharacters": [".", ":", "="]
            },
            "hoverProvider": True,
            "signatureHelpProvider": {
                "triggerCharacters": ["(", ","]
            },
            "definitionProvider": True,
            "referencesProvider": True,
            "documentHighlightProvider": random.choice([True, False]),
            "documentSymbolProvider": True,
            "workspaceSymbolProvider": True,
            "codeActionProvider": random.choice([True, False]),
            "codeLensProvider": {
                "resolveProvider": random.choice([True, False])
            },
            "documentFormattingProvider": random.choice([True, False]),
            "documentRangeFormattingProvider": random.choice([True, False]),
            "renameProvider": random.choice([True, False]),
            "documentLinkProvider": {
                "resolveProvider": random.choice([True, False])
            }
        }

    def _generate_lsp_settings(self, server_name: str) -> Dict[str, Any]:
        """Generate server-specific LSP settings."""
        settings_map = {
            "pylsp": {
                "pylsp": {
                    "plugins": {
                        "flake8": {"enabled": random.choice([True, False])},
                        "black": {"enabled": random.choice([True, False])},
                        "mypy": {"enabled": random.choice([True, False])}
                    }
                }
            },
            "rust-analyzer": {
                "rust-analyzer": {
                    "checkOnSave": {"command": "clippy"},
                    "cargo": {"features": "all"}
                }
            },
            "typescript-language-server": {
                "typescript": {
                    "preferences": {
                        "disableSuggestions": False,
                        "quotePreference": random.choice(["single", "double"])
                    }
                }
            }
        }

        return settings_map.get(server_name, {})

    def _generate_collection_mapping(self, project_spec: ProjectSpec) -> Dict[str, Any]:
        """Generate Qdrant collection mapping for the project."""
        workspace_types = ["docs", "notes", "scratchbook", "knowledge", "context", "memory"]

        collections = {}
        for workspace_type in workspace_types:
            if random.random() > 0.3:  # 70% chance to have each type
                collection_name = f"{project_spec.name}-{workspace_type}"
                collections[workspace_type] = {
                    "collection_name": collection_name,
                    "vector_count": random.randint(0, 10000),
                    "last_updated": int(time.time() - random.randint(0, 86400)),
                    "auto_sync": random.choice([True, False]),
                    "file_patterns": self._get_patterns_for_workspace_type(workspace_type),
                    "metadata_schema": {
                        "project": "keyword",
                        "file_path": "text",
                        "language": "keyword",
                        "type": "keyword",
                        "created_at": "integer",
                        "updated_at": "integer"
                    }
                }

        # Add global collections
        collections["global"] = {
            "shared_knowledge": {
                "collection_name": "shared-knowledge",
                "vector_count": random.randint(1000, 100000),
                "last_updated": int(time.time() - random.randint(0, 86400 * 7)),
                "auto_sync": True,
                "description": "Cross-project shared knowledge base"
            },
            "templates": {
                "collection_name": "code-templates",
                "vector_count": random.randint(100, 5000),
                "last_updated": int(time.time() - random.randint(0, 86400 * 30)),
                "auto_sync": True,
                "description": "Reusable code templates and patterns"
            }
        }

        return collections

    def _get_patterns_for_workspace_type(self, workspace_type: str) -> List[str]:
        """Get file patterns for specific workspace types."""
        patterns_map = {
            "docs": ["*.md", "*.rst", "*.txt", "README*"],
            "notes": ["*.md", "*.txt", "notes/*"],
            "scratchbook": ["scratch*", "temp*", "*.tmp"],
            "knowledge": ["*.md", "*.wiki", "docs/*"],
            "context": ["*.json", "*.yaml", "config/*"],
            "memory": ["*.log", "history/*", "cache/*"]
        }

        return patterns_map.get(workspace_type, ["*"])

    def generate_multi_project_workspace(self, num_projects: int = 5) -> Dict[str, Any]:
        """Generate multi-project workspace data."""
        workspace = {
            "workspace_id": str(uuid.uuid4()),
            "name": f"workspace_{random.randint(1000, 9999)}",
            "created_at": int(time.time() - random.randint(86400, 86400 * 30)),
            "projects": {},
            "shared_resources": {
                "collections": ["shared-knowledge", "code-templates", "common-patterns"],
                "configurations": {
                    "default_vector_size": random.choice([384, 768, 1536]),
                    "auto_sync_interval": random.randint(300, 3600),
                    "max_projects": random.randint(10, 100)
                }
            },
            "statistics": {
                "total_projects": num_projects,
                "total_collections": 0,
                "total_documents": 0,
                "total_size_mb": 0
            }
        }

        for i in range(num_projects):
            project_context = self.generate_project_context()
            project_id = project_context["project_name"]

            workspace["projects"][project_id] = project_context

            # Update workspace statistics
            workspace["statistics"]["total_collections"] += len(project_context["collections"])
            for collection_data in project_context["collections"].values():
                if isinstance(collection_data, dict) and "vector_count" in collection_data:
                    workspace["statistics"]["total_documents"] += collection_data["vector_count"]

            workspace["statistics"]["total_size_mb"] += project_context["structure"]["root"]["total_size_bytes"] // (1024 * 1024)

        return workspace

    def generate_language_detection_data(self) -> Dict[str, Any]:
        """Generate language detection test data."""
        detection_data = {}

        for lang_name, config in self.language_configs.items():
            samples = []
            for ext in config.extensions:
                samples.append({
                    "extension": ext,
                    "confidence": random.uniform(0.8, 1.0),
                    "file_sample": f"sample_file{ext}",
                    "content_indicators": self._generate_content_indicators(lang_name),
                    "tree_sitter_available": config.tree_sitter_grammar is not None,
                    "lsp_available": config.lsp_server is not None
                })

            detection_data[lang_name] = {
                "language": config.name,
                "samples": samples,
                "total_files_detected": random.randint(0, 1000),
                "detection_accuracy": random.uniform(0.85, 0.99)
            }

        return detection_data

    def _generate_content_indicators(self, language: str) -> List[str]:
        """Generate content indicators for language detection."""
        indicators_map = {
            "python": ["def ", "import ", "class ", "__name__", "print("],
            "rust": ["fn ", "use ", "struct ", "impl ", "match "],
            "javascript": ["function", "const ", "let ", "var ", "=>"],
            "typescript": ["interface", "type ", ": string", ": number", "export"],
            "java": ["public class", "public static", "import java", "System.out"],
            "go": ["func ", "package ", "import ", "type ", "go "],
            "cpp": ["#include", "using namespace", "int main", "std::"],
            "c": ["#include", "int main", "printf", "struct "],
            "html": ["<!DOCTYPE", "<html", "<div", "<script"],
            "css": ["{", "}", ":", ";", "px"],
            "json": ["{", "}", ":", ",", "\""],
            "yaml": [":", "-", "  ", "---"],
            "markdown": ["#", "##", "```", "*", "-"],
            "sql": ["SELECT", "FROM", "WHERE", "INSERT", "UPDATE"]
        }

        return indicators_map.get(language, [f"{language}_keyword"])

    def generate_project_comparison(self, projects: List[str]) -> Dict[str, Any]:
        """Generate comparison data between multiple projects."""
        comparison = {
            "projects": projects,
            "comparison_date": int(time.time()),
            "metrics": {},
            "similarities": {},
            "differences": {}
        }

        # Generate metrics for each project
        for project in projects:
            comparison["metrics"][project] = {
                "languages": random.randint(1, 10),
                "files": random.randint(10, 10000),
                "lines_of_code": random.randint(1000, 1000000),
                "complexity_score": random.uniform(0.1, 1.0),
                "test_coverage": random.uniform(0.0, 1.0),
                "documentation_ratio": random.uniform(0.0, 0.5),
                "last_activity": int(time.time() - random.randint(0, 86400 * 30))
            }

        # Generate similarity scores
        for i, proj1 in enumerate(projects):
            for proj2 in projects[i+1:]:
                comparison["similarities"][f"{proj1}_vs_{proj2}"] = {
                    "language_overlap": random.uniform(0.0, 1.0),
                    "structure_similarity": random.uniform(0.0, 1.0),
                    "naming_similarity": random.uniform(0.0, 1.0),
                    "dependency_overlap": random.uniform(0.0, 1.0),
                    "overall_similarity": random.uniform(0.0, 1.0)
                }

        return comparison