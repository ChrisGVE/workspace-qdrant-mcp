#!/usr/bin/env python3
"""
Sample project ingestion script for workspace-qdrant-mcp software development examples.

This script demonstrates how to automatically extract and store code documentation,
architecture decisions, and project knowledge in Qdrant collections for enhanced
Claude integration.
"""

import os
import ast
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class CodeDocumentation:
    """Represents extracted code documentation"""
    name: str
    doc_type: str  # function, class, module
    content: str
    file_path: str
    line_number: int
    parameters: List[str] = None
    return_type: str = None
    complexity: str = None


@dataclass
class ProjectMetadata:
    """Project metadata for enhanced search"""
    name: str
    version: str
    description: str
    tech_stack: List[str]
    dependencies: List[str]
    last_updated: str


class SoftwareDevelopmentIngestion:
    """
    Comprehensive project ingestion for software development workflows.
    
    This class extracts and processes various types of project documentation:
    - Code docstrings and comments
    - README and markdown files
    - Configuration files
    - Test documentation
    - Architecture decisions
    """
    
    def __init__(self, project_path: str, project_name: Optional[str] = None):
        self.project_path = Path(project_path)
        self.project_name = project_name or self.project_path.name
        self.extracted_data = {}
        
    def extract_all(self) -> Dict[str, Any]:
        """
        Extract all project documentation and metadata.
        
        Returns:
            Dict containing all extracted documentation organized by type
        """
        print(f"🔍 Starting comprehensive extraction for {self.project_name}")
        
        self.extracted_data = {
            'metadata': self._extract_project_metadata(),
            'code_docs': self._extract_code_documentation(),
            'markdown_docs': self._extract_markdown_documentation(),
            'config_files': self._extract_configuration_files(),
            'test_docs': self._extract_test_documentation(),
            'architecture': self._extract_architecture_docs(),
            'dependencies': self._extract_dependencies()
        }
        
        print(f"✅ Extraction complete. Found {len(self.extracted_data)} categories")
        return self.extracted_data
    
    def _extract_project_metadata(self) -> ProjectMetadata:
        """Extract basic project metadata"""
        print("📊 Extracting project metadata...")
        
        # Try to get version from various sources
        version = "unknown"
        description = ""
        
        # Check setup.py
        setup_py = self.project_path / "setup.py"
        if setup_py.exists():
            version, description = self._parse_setup_py(setup_py)
        
        # Check pyproject.toml
        pyproject = self.project_path / "pyproject.toml"
        if pyproject.exists() and not version != "unknown":
            version, description = self._parse_pyproject_toml(pyproject)
            
        # Check package.json for Node.js projects
        package_json = self.project_path / "package.json"
        if package_json.exists():
            version, description = self._parse_package_json(package_json)
        
        # Detect tech stack
        tech_stack = self._detect_tech_stack()
        dependencies = self._extract_dependencies_list()
        
        return ProjectMetadata(
            name=self.project_name,
            version=version,
            description=description,
            tech_stack=tech_stack,
            dependencies=dependencies,
            last_updated=datetime.now().isoformat()
        )
    
    def _extract_code_documentation(self) -> List[CodeDocumentation]:
        """Extract docstrings and comments from Python files"""
        print("🐍 Extracting Python code documentation...")
        
        docs = []
        python_files = list(self.project_path.rglob("*.py"))
        
        for file_path in python_files:
            # Skip virtual environments and build directories
            if any(skip in str(file_path) for skip in ['.venv', 'venv', '__pycache__', '.git', 'build', 'dist']):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                file_docs = self._parse_ast_for_docs(tree, file_path)
                docs.extend(file_docs)
                
            except (SyntaxError, UnicodeDecodeError) as e:
                print(f"⚠️  Warning: Could not parse {file_path}: {e}")
                continue
        
        print(f"📝 Found {len(docs)} code documentation entries")
        return docs
    
    def _parse_ast_for_docs(self, tree: ast.AST, file_path: Path) -> List[CodeDocumentation]:
        """Parse AST for documentation"""
        docs = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                doc = ast.get_docstring(node)
                if doc:
                    # Extract parameter information
                    params = [arg.arg for arg in node.args.args]
                    return_annotation = ast.unparse(node.returns) if node.returns else None
                    
                    docs.append(CodeDocumentation(
                        name=node.name,
                        doc_type='function',
                        content=doc,
                        file_path=str(file_path.relative_to(self.project_path)),
                        line_number=node.lineno,
                        parameters=params,
                        return_type=return_annotation,
                        complexity=self._calculate_complexity(node)
                    ))
            
            elif isinstance(node, ast.ClassDef):
                doc = ast.get_docstring(node)
                if doc:
                    docs.append(CodeDocumentation(
                        name=node.name,
                        doc_type='class',
                        content=doc,
                        file_path=str(file_path.relative_to(self.project_path)),
                        line_number=node.lineno
                    ))
        
        # Check for module-level docstring
        if (isinstance(tree.body[0], ast.Expr) and 
            isinstance(tree.body[0].value, ast.Str)):
            docs.append(CodeDocumentation(
                name='module',
                doc_type='module',
                content=tree.body[0].value.s,
                file_path=str(file_path.relative_to(self.project_path)),
                line_number=1
            ))
        
        return docs
    
    def _extract_markdown_documentation(self) -> List[Dict[str, Any]]:
        """Extract markdown documentation files"""
        print("📖 Extracting markdown documentation...")
        
        docs = []
        markdown_files = list(self.project_path.rglob("*.md"))
        markdown_files.extend(list(self.project_path.rglob("*.rst")))
        
        for file_path in markdown_files:
            if any(skip in str(file_path) for skip in ['.git', 'node_modules', '.venv']):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract headers for better organization
                headers = self._extract_markdown_headers(content)
                
                docs.append({
                    'file_path': str(file_path.relative_to(self.project_path)),
                    'content': content,
                    'headers': headers,
                    'file_type': file_path.suffix.lower(),
                    'size': len(content),
                    'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })
                
            except UnicodeDecodeError:
                print(f"⚠️  Warning: Could not read {file_path} (encoding issue)")
                continue
        
        print(f"📄 Found {len(docs)} documentation files")
        return docs
    
    def _extract_configuration_files(self) -> List[Dict[str, Any]]:
        """Extract configuration files"""
        print("⚙️  Extracting configuration files...")
        
        config_files = []
        config_patterns = [
            "*.json", "*.yaml", "*.yml", "*.toml", "*.ini", "*.cfg",
            ".env*", "Dockerfile*", "docker-compose*", "requirements*.txt",
            "setup.py", "setup.cfg", "pyproject.toml", "package.json"
        ]
        
        for pattern in config_patterns:
            for file_path in self.project_path.rglob(pattern):
                if any(skip in str(file_path) for skip in ['.git', 'node_modules', '.venv', '__pycache__']):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    config_files.append({
                        'file_path': str(file_path.relative_to(self.project_path)),
                        'content': content,
                        'file_type': file_path.suffix.lower() or file_path.name.lower(),
                        'size': len(content)
                    })
                    
                except (UnicodeDecodeError, IsADirectoryError):
                    continue
        
        print(f"⚙️  Found {len(config_files)} configuration files")
        return config_files
    
    def _extract_test_documentation(self) -> List[Dict[str, Any]]:
        """Extract test files and documentation"""
        print("🧪 Extracting test documentation...")
        
        test_docs = []
        test_directories = ['test', 'tests', 'spec', 'specs']
        
        for test_dir in test_directories:
            test_path = self.project_path / test_dir
            if test_path.exists():
                for file_path in test_path.rglob("*.py"):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Extract test function names and docstrings
                        test_functions = self._extract_test_functions(content)
                        
                        test_docs.append({
                            'file_path': str(file_path.relative_to(self.project_path)),
                            'content': content,
                            'test_functions': test_functions,
                            'file_type': 'test'
                        })
                        
                    except UnicodeDecodeError:
                        continue
        
        print(f"🧪 Found {len(test_docs)} test files")
        return test_docs
    
    def _extract_architecture_docs(self) -> List[Dict[str, Any]]:
        """Extract architecture documentation"""
        print("🏗️  Extracting architecture documentation...")
        
        arch_docs = []
        arch_patterns = ["*adr*", "*architecture*", "*design*", "*ADR*"]
        
        for pattern in arch_patterns:
            for file_path in self.project_path.rglob(pattern):
                if file_path.is_file() and file_path.suffix.lower() in ['.md', '.rst', '.txt']:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        arch_docs.append({
                            'file_path': str(file_path.relative_to(self.project_path)),
                            'content': content,
                            'doc_type': 'architecture',
                            'extracted_date': datetime.now().isoformat()
                        })
                        
                    except UnicodeDecodeError:
                        continue
        
        print(f"🏗️  Found {len(arch_docs)} architecture documents")
        return arch_docs
    
    def _detect_tech_stack(self) -> List[str]:
        """Detect technology stack from project files"""
        tech_stack = []
        
        # Python indicators
        if any(self.project_path.glob(pattern) for pattern in ["*.py", "requirements*.txt", "setup.py", "pyproject.toml"]):
            tech_stack.append("Python")
        
        # JavaScript/Node.js indicators
        if (self.project_path / "package.json").exists():
            tech_stack.append("JavaScript/Node.js")
        
        # Docker indicators
        if any(self.project_path.glob("Dockerfile*")) or (self.project_path / "docker-compose.yml").exists():
            tech_stack.append("Docker")
        
        # Framework detection
        if (self.project_path / "requirements.txt").exists():
            with open(self.project_path / "requirements.txt", 'r') as f:
                content = f.read().lower()
                if "django" in content:
                    tech_stack.append("Django")
                if "flask" in content:
                    tech_stack.append("Flask")
                if "fastapi" in content:
                    tech_stack.append("FastAPI")
        
        return tech_stack
    
    def _extract_dependencies_list(self) -> List[str]:
        """Extract project dependencies"""
        dependencies = []
        
        # Python requirements
        req_file = self.project_path / "requirements.txt"
        if req_file.exists():
            with open(req_file, 'r') as f:
                dependencies.extend([line.strip().split('==')[0] for line in f if line.strip() and not line.startswith('#')])
        
        return dependencies[:20]  # Limit to first 20 to avoid overwhelming
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> str:
        """Calculate function complexity (simplified)"""
        complexity_score = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try)):
                complexity_score += 1
        
        if complexity_score <= 3:
            return "low"
        elif complexity_score <= 7:
            return "medium"
        else:
            return "high"
    
    def _extract_markdown_headers(self, content: str) -> List[str]:
        """Extract markdown headers from content"""
        headers = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                headers.append(line)
        
        return headers
    
    def _extract_test_functions(self, content: str) -> List[str]:
        """Extract test function names from test file content"""
        test_functions = []
        
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    test_functions.append(node.name)
        except SyntaxError:
            pass
        
        return test_functions
    
    def _parse_setup_py(self, file_path: Path) -> tuple:
        """Parse setup.py for version and description"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            # Simplified parsing - in real implementation, you'd want more robust parsing
            version = "unknown"
            description = ""
            return version, description
        except:
            return "unknown", ""
    
    def _parse_pyproject_toml(self, file_path: Path) -> tuple:
        """Parse pyproject.toml for version and description"""
        try:
            # In real implementation, use tomli/tomllib
            return "unknown", ""
        except:
            return "unknown", ""
    
    def _parse_package_json(self, file_path: Path) -> tuple:
        """Parse package.json for version and description"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            version = data.get('version', 'unknown')
            description = data.get('description', '')
            return version, description
        except:
            return "unknown", ""
    
    def save_extraction_results(self, output_file: str = None):
        """Save extraction results to JSON file"""
        if not output_file:
            output_file = f"{self.project_name}_extraction_results.json"
        
        # Convert dataclasses to dicts
        serializable_data = {}
        for key, value in self.extracted_data.items():
            if key == 'metadata':
                serializable_data[key] = asdict(value)
            elif key == 'code_docs':
                serializable_data[key] = [asdict(doc) for doc in value]
            else:
                serializable_data[key] = value
        
        with open(output_file, 'w') as f:
            json.dump(serializable_data, f, indent=2, default=str)
        
        print(f"💾 Extraction results saved to {output_file}")


def main():
    """Example usage of the ingestion system"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract software development project documentation")
    parser.add_argument("project_path", help="Path to the project directory")
    parser.add_argument("--project-name", help="Project name (defaults to directory name)")
    parser.add_argument("--output", help="Output JSON file")
    
    args = parser.parse_args()
    
    # Initialize and run extraction
    extractor = SoftwareDevelopmentIngestion(args.project_path, args.project_name)
    results = extractor.extract_all()
    
    # Save results
    extractor.save_extraction_results(args.output)
    
    # Print summary
    print("\n📊 Extraction Summary:")
    print(f"Project: {results['metadata'].name}")
    print(f"Tech Stack: {', '.join(results['metadata'].tech_stack)}")
    print(f"Code Documentation Entries: {len(results['code_docs'])}")
    print(f"Markdown Documents: {len(results['markdown_docs'])}")
    print(f"Configuration Files: {len(results['config_files'])}")
    print(f"Test Files: {len(results['test_docs'])}")
    print(f"Architecture Documents: {len(results['architecture'])}")
    
    print(f"\n✅ Ready for ingestion into workspace-qdrant-mcp!")
    print(f"Next step: Use this data with Claude for enhanced project understanding.")


if __name__ == "__main__":
    main()