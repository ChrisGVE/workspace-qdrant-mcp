#!/usr/bin/env python3
"""
Coverage Gap Analysis Script
Analyzes Python source files to identify functions and create coverage baseline
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

# Add source to Python path
sys.path.insert(0, '/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp/src/python')

class FunctionAnalyzer(ast.NodeVisitor):
    """Analyzes Python AST to extract function definitions"""

    def __init__(self):
        self.functions = []
        self.classes = []
        self.methods = []

    def visit_FunctionDef(self, node):
        """Visit function definitions"""
        self.functions.append({
            'name': node.name,
            'line': node.lineno,
            'type': 'function',
            'is_async': isinstance(node, ast.AsyncFunctionDef),
            'decorators': [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list]
        })
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        """Visit async function definitions"""
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):
        """Visit class definitions"""
        self.classes.append({
            'name': node.name,
            'line': node.lineno,
            'type': 'class'
        })

        # Visit methods within the class
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.methods.append({
                    'name': f"{node.name}.{item.name}",
                    'line': item.lineno,
                    'type': 'method',
                    'class': node.name,
                    'is_async': isinstance(item, ast.AsyncFunctionDef),
                    'decorators': [d.id if isinstance(d, ast.Name) else str(d) for d in item.decorator_list]
                })

        self.generic_visit(node)

def analyze_python_file(file_path: Path) -> Dict:
    """Analyze a single Python file for functions and classes"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content)
        analyzer = FunctionAnalyzer()
        analyzer.visit(tree)

        return {
            'file': str(file_path),
            'functions': analyzer.functions,
            'classes': analyzer.classes,
            'methods': analyzer.methods,
            'lines': len(content.splitlines()),
            'imports': [node.names[0].name for node in ast.walk(tree)
                       if isinstance(node, ast.Import)],
            'from_imports': [node.module for node in ast.walk(tree)
                           if isinstance(node, ast.ImportFrom) and node.module]
        }

    except Exception as e:
        return {
            'file': str(file_path),
            'error': str(e),
            'functions': [],
            'classes': [],
            'methods': [],
            'lines': 0,
            'imports': [],
            'from_imports': []
        }

def analyze_source_directory(src_dir: Path) -> Dict:
    """Analyze all Python files in source directory"""
    analysis = {
        'files': {},
        'summary': {
            'total_files': 0,
            'total_functions': 0,
            'total_classes': 0,
            'total_methods': 0,
            'total_lines': 0,
            'components': defaultdict(dict)
        }
    }

    # Find all Python files
    python_files = list(src_dir.rglob("*.py"))

    # Exclude certain files
    exclude_patterns = [
        '__pycache__',
        '.egg-info',
        'test_',
        '_test.py',
        'conftest.py'
    ]

    filtered_files = []
    for file_path in python_files:
        if not any(pattern in str(file_path) for pattern in exclude_patterns):
            filtered_files.append(file_path)

    print(f"Analyzing {len(filtered_files)} Python files...")

    for file_path in filtered_files:
        relative_path = file_path.relative_to(src_dir)
        component = str(relative_path).split('/')[0]

        file_analysis = analyze_python_file(file_path)
        analysis['files'][str(relative_path)] = file_analysis

        # Update summary
        analysis['summary']['total_files'] += 1
        analysis['summary']['total_functions'] += len(file_analysis['functions'])
        analysis['summary']['total_classes'] += len(file_analysis['classes'])
        analysis['summary']['total_methods'] += len(file_analysis['methods'])
        analysis['summary']['total_lines'] += file_analysis['lines']

        # Update component summary
        comp_summary = analysis['summary']['components'][component]
        comp_summary['files'] = comp_summary.get('files', 0) + 1
        comp_summary['functions'] = comp_summary.get('functions', 0) + len(file_analysis['functions'])
        comp_summary['classes'] = comp_summary.get('classes', 0) + len(file_analysis['classes'])
        comp_summary['methods'] = comp_summary.get('methods', 0) + len(file_analysis['methods'])
        comp_summary['lines'] = comp_summary.get('lines', 0) + file_analysis['lines']

    return analysis

def identify_mcp_tools(analysis: Dict) -> List[Dict]:
    """Identify MCP tool functions from analysis"""
    mcp_tools = []

    for file_path, file_analysis in analysis['files'].items():
        if 'tools/' in file_path or file_path.endswith('tools.py'):
            for func in file_analysis['functions']:
                # Check for MCP tool patterns
                if any(decorator in ['app.tool', 'tool', 'mcp_tool'] for decorator in func.get('decorators', [])):
                    mcp_tools.append({
                        'file': file_path,
                        'function': func['name'],
                        'line': func['line'],
                        'type': 'mcp_tool'
                    })

            for method in file_analysis['methods']:
                if any(decorator in ['app.tool', 'tool', 'mcp_tool'] for decorator in method.get('decorators', [])):
                    mcp_tools.append({
                        'file': file_path,
                        'function': method['name'],
                        'line': method['line'],
                        'type': 'mcp_tool_method'
                    })

    return mcp_tools

def generate_coverage_report(analysis: Dict) -> str:
    """Generate comprehensive coverage gap analysis report"""
    report = []
    report.append("# Test Coverage Gap Analysis Report")
    report.append(f"Generated: {Path().cwd()}")
    report.append("")

    # Summary statistics
    summary = analysis['summary']
    report.append("## Summary Statistics")
    report.append(f"- **Total Python Files**: {summary['total_files']}")
    report.append(f"- **Total Functions**: {summary['total_functions']}")
    report.append(f"- **Total Classes**: {summary['total_classes']}")
    report.append(f"- **Total Methods**: {summary['total_methods']}")
    report.append(f"- **Total Lines of Code**: {summary['total_lines']}")
    report.append("")

    # Component breakdown
    report.append("## Component Analysis")
    for component, comp_data in summary['components'].items():
        report.append(f"### {component}")
        report.append(f"- Files: {comp_data['files']}")
        report.append(f"- Functions: {comp_data['functions']}")
        report.append(f"- Classes: {comp_data['classes']}")
        report.append(f"- Methods: {comp_data['methods']}")
        report.append(f"- Lines: {comp_data['lines']}")
        report.append("")

    # MCP Tools identification
    mcp_tools = identify_mcp_tools(analysis)
    report.append("## MCP Tools Identified")
    report.append(f"Found {len(mcp_tools)} MCP tool functions:")
    for tool in mcp_tools:
        report.append(f"- `{tool['function']}` in {tool['file']} (line {tool['line']})")
    report.append("")

    # High-priority files for testing
    report.append("## High-Priority Files for Test Coverage")
    priority_patterns = [
        'server.py',
        'tools/',
        'core/',
        'memory',
        'search',
        'client'
    ]

    priority_files = []
    for file_path, file_analysis in analysis['files'].items():
        if any(pattern in file_path for pattern in priority_patterns):
            func_count = len(file_analysis['functions']) + len(file_analysis['methods'])
            priority_files.append({
                'file': file_path,
                'functions': func_count,
                'lines': file_analysis['lines']
            })

    # Sort by function count
    priority_files.sort(key=lambda x: x['functions'], reverse=True)

    for pf in priority_files[:20]:  # Top 20
        report.append(f"- `{pf['file']}`: {pf['functions']} functions, {pf['lines']} lines")
    report.append("")

    # Coverage gap recommendations
    report.append("## Coverage Gap Recommendations")
    report.append("### Immediate Priority (Phase 1)")
    report.append("1. **MCP Server Tools** - 11 tools across multiple files")
    report.append("2. **Core Infrastructure** - hybrid_search.py, memory.py, client.py")
    report.append("3. **State Management** - collection management and validation")
    report.append("")

    report.append("### Secondary Priority (Phase 2)")
    report.append("1. **CLI Utilities** - wqm commands and workflows")
    report.append("2. **Context Injector** - document ingestion and processing")
    report.append("3. **Configuration System** - config validation and management")
    report.append("")

    report.append("### Testing Strategy")
    report.append("1. **Unit Tests**: Focus on individual functions and methods")
    report.append("2. **Integration Tests**: MCP tool interactions and workflows")
    report.append("3. **End-to-End Tests**: Complete user scenarios")
    report.append("4. **Performance Tests**: Search and ingestion benchmarks")
    report.append("")

    # Detailed file analysis
    report.append("## Detailed File Analysis")
    for file_path, file_analysis in analysis['files'].items():
        if file_analysis.get('error'):
            continue

        func_count = len(file_analysis['functions']) + len(file_analysis['methods'])
        if func_count > 0:
            report.append(f"### {file_path}")
            report.append(f"- Functions: {len(file_analysis['functions'])}")
            report.append(f"- Methods: {len(file_analysis['methods'])}")
            report.append(f"- Classes: {len(file_analysis['classes'])}")
            report.append(f"- Lines: {file_analysis['lines']}")

            if file_analysis['functions']:
                report.append("- Functions:")
                for func in file_analysis['functions']:
                    decorators = f" @{','.join(func['decorators'])}" if func['decorators'] else ""
                    async_marker = "async " if func['is_async'] else ""
                    report.append(f"  - `{async_marker}{func['name']}()` (line {func['line']}){decorators}")

            if file_analysis['methods']:
                report.append("- Methods:")
                for method in file_analysis['methods']:
                    decorators = f" @{','.join(method['decorators'])}" if method['decorators'] else ""
                    async_marker = "async " if method['is_async'] else ""
                    report.append(f"  - `{async_marker}{method['name']}()` (line {method['line']}){decorators}")

            report.append("")

    return "\n".join(report)

def main():
    """Main analysis function"""
    src_dir = Path('/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp/src/python')

    if not src_dir.exists():
        print(f"Source directory not found: {src_dir}")
        return

    print("Starting comprehensive source code analysis...")
    analysis = analyze_source_directory(src_dir)

    print("\nGenerating coverage gap report...")
    report = generate_coverage_report(analysis)

    # Save report
    report_file = Path('20250921-1502_coverage_gap_analysis_report.md')
    with open(report_file, 'w') as f:
        f.write(report)

    print(f"\nCoverage gap analysis complete!")
    print(f"Report saved to: {report_file}")
    print(f"Total functions identified: {analysis['summary']['total_functions']}")
    print(f"Total methods identified: {analysis['summary']['total_methods']}")
    print(f"Total callable units: {analysis['summary']['total_functions'] + analysis['summary']['total_methods']}")

if __name__ == "__main__":
    main()