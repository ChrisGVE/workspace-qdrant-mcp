#!/usr/bin/env python3
"""
Comprehensive Coverage Gap Analyzer
Identifies specific uncovered code and generates targeted test recommendations.
"""

import json
import ast
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple
import re


class CoverageGapAnalyzer:
    """Analyzes coverage gaps and provides specific testing recommendations"""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.coverage_data = None
        self.load_coverage_data()

    def load_coverage_data(self):
        """Load the latest coverage data"""
        try:
            # Generate fresh coverage data
            subprocess.run([
                'uv', 'run', 'pytest', '--cov=src', '--cov-report=json',
                '--tb=short', '-q'
            ], capture_output=True, text=True, cwd=self.project_root)

            coverage_json_path = self.project_root / "coverage.json"
            if coverage_json_path.exists():
                with open(coverage_json_path) as f:
                    self.coverage_data = json.load(f)
            else:
                print("❌ No coverage data available")

        except Exception as e:
            print(f"❌ Error loading coverage data: {e}")

    def get_file_analysis(self, filepath: str) -> Dict:
        """Analyze a specific file's coverage gaps"""
        if not self.coverage_data:
            return {}

        files = self.coverage_data.get('files', {})
        file_data = files.get(filepath, {})

        if not file_data:
            return {}

        missing_lines = file_data.get('missing_lines', [])
        executed_lines = file_data.get('executed_lines', [])
        summary = file_data.get('summary', {})

        try:
            full_path = self.project_root / filepath
            if not full_path.exists():
                return {}

            with open(full_path) as f:
                content = f.read()
                lines = content.split('\n')

            # Parse AST to understand code structure
            try:
                tree = ast.parse(content)
                functions = self.extract_functions(tree)
                classes = self.extract_classes(tree)
            except:
                functions = []
                classes = []

            return {
                'filepath': filepath,
                'total_lines': len(lines),
                'missing_lines': missing_lines,
                'executed_lines': executed_lines,
                'coverage_percent': summary.get('percent_covered', 0),
                'missing_statements': summary.get('missing_lines', 0),
                'functions': functions,
                'classes': classes,
                'content_lines': lines,
                'uncovered_functions': self.find_uncovered_functions(functions, missing_lines),
                'uncovered_classes': self.find_uncovered_classes(classes, missing_lines)
            }

        except Exception as e:
            print(f"Error analyzing {filepath}: {e}")
            return {}

    def extract_functions(self, tree: ast.AST) -> List[Dict]:
        """Extract function definitions from AST"""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append({
                    'name': node.name,
                    'line_start': node.lineno,
                    'line_end': getattr(node, 'end_lineno', node.lineno),
                    'is_async': isinstance(node, ast.AsyncFunctionDef),
                    'args': [arg.arg for arg in node.args.args],
                    'decorators': [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list]
                })
        return functions

    def extract_classes(self, tree: ast.AST) -> List[Dict]:
        """Extract class definitions from AST"""
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append({
                            'name': item.name,
                            'line_start': item.lineno,
                            'line_end': getattr(item, 'end_lineno', item.lineno)
                        })

                classes.append({
                    'name': node.name,
                    'line_start': node.lineno,
                    'line_end': getattr(node, 'end_lineno', node.lineno),
                    'methods': methods,
                    'bases': [base.id if isinstance(base, ast.Name) else str(base) for base in node.bases]
                })
        return classes

    def find_uncovered_functions(self, functions: List[Dict], missing_lines: List[int]) -> List[Dict]:
        """Find functions that have uncovered lines"""
        uncovered = []
        for func in functions:
            func_lines = set(range(func['line_start'], func['line_end'] + 1))
            missing_in_func = func_lines.intersection(set(missing_lines))
            if missing_in_func:
                uncovered.append({
                    **func,
                    'uncovered_lines': sorted(list(missing_in_func)),
                    'coverage_percent': (len(func_lines) - len(missing_in_func)) / len(func_lines) * 100
                })
        return uncovered

    def find_uncovered_classes(self, classes: List[Dict], missing_lines: List[int]) -> List[Dict]:
        """Find classes that have uncovered lines"""
        uncovered = []
        for cls in classes:
            cls_lines = set(range(cls['line_start'], cls['line_end'] + 1))
            missing_in_cls = cls_lines.intersection(set(missing_lines))
            if missing_in_cls:
                uncovered.append({
                    **cls,
                    'uncovered_lines': sorted(list(missing_in_cls)),
                    'coverage_percent': (len(cls_lines) - len(missing_in_cls)) / len(cls_lines) * 100
                })
        return uncovered

    def generate_test_code_suggestions(self, file_analysis: Dict) -> List[str]:
        """Generate specific test code suggestions"""
        suggestions = []
        filepath = file_analysis['filepath']
        uncovered_functions = file_analysis['uncovered_functions']
        uncovered_classes = file_analysis['uncovered_classes']

        # Generate import statement
        module_path = filepath.replace('src/python/', '').replace('/', '.').replace('.py', '')
        suggestions.append(f"# Test suggestions for {filepath}")
        suggestions.append(f"import pytest")
        suggestions.append(f"from {module_path} import *")
        suggestions.append("")

        # Generate function tests
        for func in uncovered_functions[:5]:  # Top 5 functions
            test_name = f"test_{func['name']}"
            suggestions.append(f"def {test_name}():")
            suggestions.append(f"    \"\"\"Test {func['name']} function\"\"\"")

            if func['is_async']:
                suggestions.append(f"    # Async function test")
                suggestions.append(f"    # TODO: Add async test implementation")
            else:
                suggestions.append(f"    # TODO: Add test implementation")

            if 'error' in func['name'].lower() or 'exception' in func['name'].lower():
                suggestions.append(f"    # TODO: Test error handling scenarios")

            suggestions.append(f"    pass")
            suggestions.append("")

        # Generate class tests
        for cls in uncovered_classes[:3]:  # Top 3 classes
            test_class_name = f"Test{cls['name']}"
            suggestions.append(f"class {test_class_name}:")
            suggestions.append(f"    \"\"\"Test {cls['name']} class\"\"\"")
            suggestions.append("")

            suggestions.append(f"    def test_init(self):")
            suggestions.append(f"        \"\"\"Test {cls['name']} initialization\"\"\"")
            suggestions.append(f"        # TODO: Add initialization test")
            suggestions.append(f"        pass")
            suggestions.append("")

            for method in cls['methods'][:3]:  # Top 3 methods
                suggestions.append(f"    def test_{method['name']}(self):")
                suggestions.append(f"        \"\"\"Test {method['name']} method\"\"\"")
                suggestions.append(f"        # TODO: Add method test")
                suggestions.append(f"        pass")
                suggestions.append("")

        return suggestions

    def analyze_top_priority_files(self, num_files: int = 10) -> List[Dict]:
        """Analyze top priority files needing coverage"""
        if not self.coverage_data:
            return []

        files = self.coverage_data.get('files', {})
        priority_files = []

        for filepath, file_data in files.items():
            summary = file_data.get('summary', {})
            missing_lines = file_data.get('missing_lines', [])

            # Calculate priority score
            coverage_percent = summary.get('percent_covered', 0)
            total_lines = summary.get('num_statements', 0)
            missing_count = len(missing_lines)

            # Higher priority for core files
            core_modules = ['server.py', 'client.py', 'memory.py', 'hybrid_search.py']
            is_core = any(module in filepath for module in core_modules)

            priority_score = (100 - coverage_percent) * 0.4
            priority_score += missing_count * 0.3
            priority_score += total_lines * 0.2
            priority_score += 50 if is_core else 0

            analysis = self.get_file_analysis(filepath)
            if analysis:
                analysis['priority_score'] = priority_score
                priority_files.append(analysis)

        # Sort by priority score
        priority_files.sort(key=lambda x: x['priority_score'], reverse=True)
        return priority_files[:num_files]

    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive gap analysis report"""
        if not self.coverage_data:
            return "❌ No coverage data available"

        totals = self.coverage_data.get('totals', {})
        priority_files = self.analyze_top_priority_files()

        report = f"""
🔍 COMPREHENSIVE COVERAGE GAP ANALYSIS
=====================================

OVERVIEW:
📊 Total Coverage: {totals.get('percent_covered', 0):.2f}%
✅ Covered Lines: {totals.get('covered_lines', 0)}
❌ Missing Lines: {totals.get('missing_lines', 0)}
📁 Total Files: {len(self.coverage_data.get('files', {}))}

TOP PRIORITY FILES FOR IMMEDIATE TESTING:
"""

        for i, file_analysis in enumerate(priority_files[:5], 1):
            report += f"""
{i}. {file_analysis['filepath']}
   📊 Coverage: {file_analysis['coverage_percent']:.1f}%
   📝 Missing Lines: {len(file_analysis['missing_lines'])}
   🎯 Priority Score: {file_analysis['priority_score']:.1f}

   Uncovered Functions ({len(file_analysis['uncovered_functions'])}):"""

            for func in file_analysis['uncovered_functions'][:3]:
                report += f"\n   • {func['name']}() - {func['coverage_percent']:.1f}% covered"

            if file_analysis['uncovered_classes']:
                report += f"\n   \n   Uncovered Classes ({len(file_analysis['uncovered_classes'])}):"
                for cls in file_analysis['uncovered_classes'][:2]:
                    report += f"\n   • {cls['name']} - {cls['coverage_percent']:.1f}% covered"

        report += "\n\n🎯 SPECIFIC ACTIONS NEEDED:\n"

        for i, file_analysis in enumerate(priority_files[:3], 1):
            report += f"\n{i}. Create test file for {file_analysis['filepath']}:"
            test_suggestions = self.generate_test_code_suggestions(file_analysis)
            report += f"\n   📝 Lines of test code needed: ~{len(test_suggestions)}"
            report += f"\n   🎯 Focus on {len(file_analysis['uncovered_functions'])} uncovered functions"

        return report

    def export_test_templates(self, output_dir: str = "test_templates"):
        """Export test templates for priority files"""
        output_path = self.project_root / output_dir
        output_path.mkdir(exist_ok=True)

        priority_files = self.analyze_top_priority_files(5)

        for file_analysis in priority_files:
            filepath = file_analysis['filepath']
            test_filename = f"test_{Path(filepath).name}"

            test_code = self.generate_test_code_suggestions(file_analysis)
            test_content = "\n".join(test_code)

            test_file_path = output_path / test_filename
            with open(test_file_path, 'w') as f:
                f.write(test_content)

            print(f"📝 Generated test template: {test_file_path}")

        print(f"✅ Test templates exported to {output_path}")


def main():
    """Main analysis execution"""
    analyzer = CoverageGapAnalyzer()

    print("🔍 Running comprehensive coverage gap analysis...")
    report = analyzer.generate_comprehensive_report()
    print(report)

    print("\n📝 Generating test templates...")
    analyzer.export_test_templates()

    print("\n✅ Coverage gap analysis complete!")


if __name__ == "__main__":
    main()