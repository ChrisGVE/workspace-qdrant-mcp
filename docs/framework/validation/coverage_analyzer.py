"""Documentation coverage analyzer for measuring completeness."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union
from pathlib import Path
import json

try:
    from ..generators.ast_parser import DocumentationNode, MemberType, PythonASTParser
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from generators.ast_parser import DocumentationNode, MemberType, PythonASTParser


@dataclass
class CoverageStats:
    """Statistics for documentation coverage."""
    total_items: int = 0
    documented_items: int = 0
    missing_docstring: int = 0
    missing_parameters: int = 0
    missing_returns: int = 0
    missing_examples: int = 0
    coverage_percentage: float = 0.0

    def calculate_percentage(self):
        """Calculate and update coverage percentage."""
        if self.total_items > 0:
            self.coverage_percentage = (self.documented_items / self.total_items) * 100
        else:
            self.coverage_percentage = 100.0


@dataclass
class MemberCoverage:
    """Coverage information for a code member."""
    name: str
    member_type: MemberType
    has_docstring: bool = False
    has_parameters_documented: bool = True  # True by default for non-functions
    has_return_documented: bool = True  # True by default for non-functions
    has_examples: bool = False
    issues: List[str] = field(default_factory=list)
    source_file: Optional[str] = None
    line_number: Optional[int] = None

    @property
    def is_fully_documented(self) -> bool:
        """Check if the member is fully documented."""
        return (self.has_docstring and
                self.has_parameters_documented and
                self.has_return_documented)


@dataclass
class FileCoverage:
    """Coverage information for a file."""
    file_path: str
    members: List[MemberCoverage] = field(default_factory=list)
    stats: CoverageStats = field(default_factory=CoverageStats)

    def calculate_stats(self):
        """Calculate coverage statistics for this file."""
        self.stats = CoverageStats()
        self.stats.total_items = len(self.members)

        for member in self.members:
            if member.has_docstring:
                self.stats.documented_items += 1
            else:
                self.stats.missing_docstring += 1

            if not member.has_parameters_documented:
                self.stats.missing_parameters += 1

            if not member.has_return_documented:
                self.stats.missing_returns += 1

            if not member.has_examples:
                self.stats.missing_examples += 1

        self.stats.calculate_percentage()


@dataclass
class ProjectCoverage:
    """Coverage information for an entire project."""
    project_path: str
    files: List[FileCoverage] = field(default_factory=list)
    overall_stats: CoverageStats = field(default_factory=CoverageStats)

    def calculate_overall_stats(self):
        """Calculate overall coverage statistics."""
        self.overall_stats = CoverageStats()

        for file_coverage in self.files:
            self.overall_stats.total_items += file_coverage.stats.total_items
            self.overall_stats.documented_items += file_coverage.stats.documented_items
            self.overall_stats.missing_docstring += file_coverage.stats.missing_docstring
            self.overall_stats.missing_parameters += file_coverage.stats.missing_parameters
            self.overall_stats.missing_returns += file_coverage.stats.missing_returns
            self.overall_stats.missing_examples += file_coverage.stats.missing_examples

        self.overall_stats.calculate_percentage()


class DocumentationCoverageAnalyzer:
    """Analyzes documentation coverage for Python codebases."""

    def __init__(self, require_examples: bool = False,
                 require_return_docs: bool = True,
                 require_param_docs: bool = True,
                 include_private: bool = False):
        """Initialize the coverage analyzer.

        Args:
            require_examples: Whether to require examples in docstrings
            require_return_docs: Whether to require return documentation for functions
            require_param_docs: Whether to require parameter documentation for functions
            include_private: Whether to analyze private members
        """
        self.require_examples = require_examples
        self.require_return_docs = require_return_docs
        self.require_param_docs = require_param_docs
        self.include_private = include_private
        self.parser = PythonASTParser(include_private=include_private)

    def analyze_file(self, file_path: Union[str, Path]) -> FileCoverage:
        """Analyze documentation coverage for a single file.

        Args:
            file_path: Path to the Python file to analyze

        Returns:
            FileCoverage object with analysis results

        Raises:
            FileNotFoundError: If the file doesn't exist
            SyntaxError: If the file has syntax errors
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            module = self.parser.parse_file(file_path)
        except SyntaxError as e:
            raise SyntaxError(f"Syntax error in {file_path}: {e}")

        file_coverage = FileCoverage(file_path=str(file_path))

        # Analyze module itself
        if module.name != '__main__':  # Skip if it's a script
            module_coverage = self._analyze_member(module)
            if module_coverage:
                file_coverage.members.append(module_coverage)

        # Analyze all members recursively
        self._analyze_members_recursive(module, file_coverage.members)

        file_coverage.calculate_stats()
        return file_coverage

    def analyze_directory(self, directory_path: Union[str, Path],
                         recursive: bool = True) -> ProjectCoverage:
        """Analyze documentation coverage for a directory.

        Args:
            directory_path: Path to the directory to analyze
            recursive: Whether to analyze subdirectories recursively

        Returns:
            ProjectCoverage object with analysis results

        Raises:
            NotADirectoryError: If the directory doesn't exist
        """
        directory_path = Path(directory_path)
        if not directory_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory_path}")

        project_coverage = ProjectCoverage(project_path=str(directory_path))

        pattern = "**/*.py" if recursive else "*.py"
        for py_file in directory_path.glob(pattern):
            if py_file.name.startswith('.'):
                continue

            try:
                file_coverage = self.analyze_file(py_file)
                project_coverage.files.append(file_coverage)
            except (SyntaxError, UnicodeDecodeError) as e:
                # Create a file coverage with error information
                file_coverage = FileCoverage(file_path=str(py_file))
                file_coverage.members = [MemberCoverage(
                    name="<parse_error>",
                    member_type=MemberType.MODULE,
                    issues=[f"Parse error: {e}"]
                )]
                project_coverage.files.append(file_coverage)

        project_coverage.calculate_overall_stats()
        return project_coverage

    def _analyze_members_recursive(self, node: DocumentationNode,
                                 members: List[MemberCoverage]) -> None:
        """Recursively analyze all members of a node."""
        for child in node.children:
            member_coverage = self._analyze_member(child)
            if member_coverage:
                members.append(member_coverage)

            # Recursively analyze children (e.g., methods in classes)
            if child.children:
                self._analyze_members_recursive(child, members)

    def _analyze_member(self, node: DocumentationNode) -> Optional[MemberCoverage]:
        """Analyze a single member for documentation coverage.

        Args:
            node: DocumentationNode to analyze

        Returns:
            MemberCoverage object or None if member should be skipped
        """
        # Skip if private member and not including private
        if not self.include_private and node.is_private:
            return None

        member_coverage = MemberCoverage(
            name=node.name,
            member_type=node.member_type,
            source_file=node.source_file,
            line_number=node.line_number
        )

        # Check docstring presence
        member_coverage.has_docstring = bool(node.docstring and node.docstring.strip())
        if not member_coverage.has_docstring:
            member_coverage.issues.append("Missing docstring")

        # Check parameter documentation (for functions and methods)
        if node.member_type in [MemberType.FUNCTION, MemberType.METHOD]:
            member_coverage.has_parameters_documented = self._check_parameter_docs(node)
            if not member_coverage.has_parameters_documented:
                member_coverage.issues.append("Missing parameter documentation")

            # Check return documentation
            member_coverage.has_return_documented = self._check_return_docs(node)
            if not member_coverage.has_return_documented:
                member_coverage.issues.append("Missing return documentation")

        # Check examples (if required)
        if self.require_examples:
            member_coverage.has_examples = bool(node.examples)
            if not member_coverage.has_examples:
                member_coverage.issues.append("Missing examples")

        return member_coverage

    def _check_parameter_docs(self, node: DocumentationNode) -> bool:
        """Check if function/method parameters are documented."""
        if not self.require_param_docs:
            return True

        if not node.parameters:
            return True  # No parameters to document

        # Get documented parameters from docstring metadata
        documented_params = set()
        if node.metadata and 'parameters' in node.metadata:
            documented_params = set(node.metadata['parameters'].keys())

        # Check if all parameters (except 'self' and 'cls') are documented
        required_params = set()
        for param in node.parameters:
            if param.name not in ['self', 'cls']:
                required_params.add(param.name)

        return required_params.issubset(documented_params)

    def _check_return_docs(self, node: DocumentationNode) -> bool:
        """Check if function/method return value is documented."""
        if not self.require_return_docs:
            return True

        # If function has no return annotation and returns None, it's optional
        if not node.return_annotation or node.return_annotation == 'None':
            return True

        # Check if return is documented
        return bool(node.return_description or
                   (node.metadata and node.metadata.get('returns')))

    def generate_report(self, coverage: Union[FileCoverage, ProjectCoverage],
                       output_format: str = 'text') -> str:
        """Generate a coverage report.

        Args:
            coverage: Coverage data to report on
            output_format: Output format ('text', 'json', 'html')

        Returns:
            Formatted coverage report as string
        """
        if output_format == 'json':
            return self._generate_json_report(coverage)
        elif output_format == 'html':
            return self._generate_html_report(coverage)
        else:
            return self._generate_text_report(coverage)

    def _generate_text_report(self, coverage: Union[FileCoverage, ProjectCoverage]) -> str:
        """Generate a text-based coverage report."""
        lines = []

        if isinstance(coverage, ProjectCoverage):
            lines.append("Documentation Coverage Report")
            lines.append("=" * 40)
            lines.append(f"Project: {coverage.project_path}")
            lines.append("")

            stats = coverage.overall_stats
            lines.append(f"Overall Coverage: {stats.coverage_percentage:.1f}%")
            lines.append(f"Total Items: {stats.total_items}")
            lines.append(f"Documented: {stats.documented_items}")
            lines.append(f"Missing Docstring: {stats.missing_docstring}")
            lines.append(f"Missing Parameters: {stats.missing_parameters}")
            lines.append(f"Missing Returns: {stats.missing_returns}")
            lines.append("")

            # File-by-file breakdown
            lines.append("File Coverage:")
            lines.append("-" * 20)
            for file_coverage in coverage.files:
                lines.append(f"{file_coverage.file_path}: {file_coverage.stats.coverage_percentage:.1f}%")

            # Detailed issues
            lines.append("")
            lines.append("Issues by File:")
            lines.append("-" * 20)
            for file_coverage in coverage.files:
                issues_found = False
                for member in file_coverage.members:
                    if member.issues:
                        if not issues_found:
                            lines.append(f"\n{file_coverage.file_path}:")
                            issues_found = True
                        lines.append(f"  {member.name} ({member.member_type.value}): {', '.join(member.issues)}")

        else:  # FileCoverage
            lines.append(f"File Coverage Report: {coverage.file_path}")
            lines.append("=" * 40)

            stats = coverage.stats
            lines.append(f"Coverage: {stats.coverage_percentage:.1f}%")
            lines.append(f"Total Items: {stats.total_items}")
            lines.append(f"Documented: {stats.documented_items}")
            lines.append("")

            # Member details
            for member in coverage.members:
                status = "✓" if member.is_fully_documented else "✗"
                lines.append(f"{status} {member.name} ({member.member_type.value})")
                if member.issues:
                    for issue in member.issues:
                        lines.append(f"    - {issue}")

        return "\n".join(lines)

    def _generate_json_report(self, coverage: Union[FileCoverage, ProjectCoverage]) -> str:
        """Generate a JSON coverage report."""
        if isinstance(coverage, ProjectCoverage):
            data = {
                'project_path': coverage.project_path,
                'overall_stats': {
                    'total_items': coverage.overall_stats.total_items,
                    'documented_items': coverage.overall_stats.documented_items,
                    'coverage_percentage': coverage.overall_stats.coverage_percentage,
                    'missing_docstring': coverage.overall_stats.missing_docstring,
                    'missing_parameters': coverage.overall_stats.missing_parameters,
                    'missing_returns': coverage.overall_stats.missing_returns,
                    'missing_examples': coverage.overall_stats.missing_examples
                },
                'files': []
            }

            for file_coverage in coverage.files:
                file_data = {
                    'file_path': file_coverage.file_path,
                    'stats': {
                        'total_items': file_coverage.stats.total_items,
                        'documented_items': file_coverage.stats.documented_items,
                        'coverage_percentage': file_coverage.stats.coverage_percentage
                    },
                    'members': []
                }

                for member in file_coverage.members:
                    member_data = {
                        'name': member.name,
                        'type': member.member_type.value,
                        'has_docstring': member.has_docstring,
                        'has_parameters_documented': member.has_parameters_documented,
                        'has_return_documented': member.has_return_documented,
                        'has_examples': member.has_examples,
                        'is_fully_documented': member.is_fully_documented,
                        'issues': member.issues,
                        'line_number': member.line_number
                    }
                    file_data['members'].append(member_data)

                data['files'].append(file_data)

        else:  # FileCoverage
            data = {
                'file_path': coverage.file_path,
                'stats': {
                    'total_items': coverage.stats.total_items,
                    'documented_items': coverage.stats.documented_items,
                    'coverage_percentage': coverage.stats.coverage_percentage
                },
                'members': []
            }

            for member in coverage.members:
                member_data = {
                    'name': member.name,
                    'type': member.member_type.value,
                    'has_docstring': member.has_docstring,
                    'has_parameters_documented': member.has_parameters_documented,
                    'has_return_documented': member.has_return_documented,
                    'has_examples': member.has_examples,
                    'is_fully_documented': member.is_fully_documented,
                    'issues': member.issues,
                    'line_number': member.line_number
                }
                data['members'].append(member_data)

        return json.dumps(data, indent=2)

    def _generate_html_report(self, coverage: Union[FileCoverage, ProjectCoverage]) -> str:
        """Generate an HTML coverage report."""
        html = [
            '<!DOCTYPE html>',
            '<html>',
            '<head>',
            '<title>Documentation Coverage Report</title>',
            '<style>',
            'body { font-family: Arial, sans-serif; margin: 40px; }',
            '.stats { background: #f5f5f5; padding: 20px; border-radius: 5px; margin: 20px 0; }',
            '.file { margin: 20px 0; border-left: 3px solid #ccc; padding-left: 20px; }',
            '.member { margin: 10px 0; padding: 10px; border: 1px solid #eee; }',
            '.documented { border-left-color: #4caf50; }',
            '.undocumented { border-left-color: #f44336; }',
            '.issues { color: #f44336; font-size: 0.9em; }',
            '.coverage { font-weight: bold; }',
            '</style>',
            '</head>',
            '<body>'
        ]

        if isinstance(coverage, ProjectCoverage):
            html.extend([
                '<h1>Documentation Coverage Report</h1>',
                f'<p>Project: {coverage.project_path}</p>',
                '<div class="stats">',
                f'<div class="coverage">Overall Coverage: {coverage.overall_stats.coverage_percentage:.1f}%</div>',
                f'<div>Total Items: {coverage.overall_stats.total_items}</div>',
                f'<div>Documented: {coverage.overall_stats.documented_items}</div>',
                '</div>'
            ])

            for file_coverage in coverage.files:
                html.extend([
                    '<div class="file">',
                    f'<h2>{file_coverage.file_path}</h2>',
                    f'<div class="coverage">Coverage: {file_coverage.stats.coverage_percentage:.1f}%</div>'
                ])

                for member in file_coverage.members:
                    css_class = "documented" if member.is_fully_documented else "undocumented"
                    html.extend([
                        f'<div class="member {css_class}">',
                        f'<strong>{member.name}</strong> ({member.member_type.value})'
                    ])

                    if member.issues:
                        html.append('<div class="issues">Issues: ' + ', '.join(member.issues) + '</div>')

                    html.append('</div>')

                html.append('</div>')

        else:  # FileCoverage
            html.extend([
                f'<h1>File Coverage: {coverage.file_path}</h1>',
                '<div class="stats">',
                f'<div class="coverage">Coverage: {coverage.stats.coverage_percentage:.1f}%</div>',
                f'<div>Total Items: {coverage.stats.total_items}</div>',
                f'<div>Documented: {coverage.stats.documented_items}</div>',
                '</div>'
            ])

            for member in coverage.members:
                css_class = "documented" if member.is_fully_documented else "undocumented"
                html.extend([
                    f'<div class="member {css_class}">',
                    f'<strong>{member.name}</strong> ({member.member_type.value})'
                ])

                if member.issues:
                    html.append('<div class="issues">Issues: ' + ', '.join(member.issues) + '</div>')

                html.append('</div>')

        html.extend(['</body>', '</html>'])
        return '\n'.join(html)

    def find_undocumented_members(self, coverage: Union[FileCoverage, ProjectCoverage]) -> List[MemberCoverage]:
        """Find all undocumented members.

        Args:
            coverage: Coverage data to analyze

        Returns:
            List of undocumented MemberCoverage objects
        """
        undocumented = []

        if isinstance(coverage, ProjectCoverage):
            for file_coverage in coverage.files:
                for member in file_coverage.members:
                    if not member.is_fully_documented:
                        undocumented.append(member)
        else:  # FileCoverage
            for member in coverage.members:
                if not member.is_fully_documented:
                    undocumented.append(member)

        return undocumented

    def meets_threshold(self, coverage: Union[FileCoverage, ProjectCoverage],
                       threshold: float) -> bool:
        """Check if coverage meets a specified threshold.

        Args:
            coverage: Coverage data to check
            threshold: Minimum coverage percentage required

        Returns:
            True if coverage meets or exceeds threshold
        """
        if isinstance(coverage, ProjectCoverage):
            return coverage.overall_stats.coverage_percentage >= threshold
        else:
            return coverage.stats.coverage_percentage >= threshold