"""Documentation quality checker for validating documentation standards."""

import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union, Tuple
from pathlib import Path
from enum import Enum

try:
    from ..generators.ast_parser import DocumentationNode, MemberType
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from generators.ast_parser import DocumentationNode, MemberType


class QualityIssueType(Enum):
    """Types of quality issues that can be detected."""
    LENGTH_TOO_SHORT = "length_too_short"
    LENGTH_TOO_LONG = "length_too_long"
    MISSING_PUNCTUATION = "missing_punctuation"
    POOR_GRAMMAR = "poor_grammar"
    INCONSISTENT_STYLE = "inconsistent_style"
    MISSING_EXAMPLES = "missing_examples"
    BROKEN_CODE_EXAMPLE = "broken_code_example"
    UNCLEAR_DESCRIPTION = "unclear_description"
    INCONSISTENT_FORMATTING = "inconsistent_formatting"
    MISSING_TYPE_INFO = "missing_type_info"
    OUTDATED_INFO = "outdated_info"
    SPELLING_ERROR = "spelling_error"


@dataclass
class QualityIssue:
    """Represents a documentation quality issue."""
    issue_type: QualityIssueType
    message: str
    severity: str = "warning"  # "error", "warning", "info"
    line_number: Optional[int] = None
    suggestion: Optional[str] = None
    context: Optional[str] = None


@dataclass
class QualityReport:
    """Quality assessment report for a code member."""
    member_name: str
    member_type: MemberType
    quality_score: float = 0.0  # 0-100
    issues: List[QualityIssue] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    source_file: Optional[str] = None
    line_number: Optional[int] = None

    def add_issue(self, issue_type: QualityIssueType, message: str,
                 severity: str = "warning", suggestion: str = None):
        """Add a quality issue to the report."""
        issue = QualityIssue(
            issue_type=issue_type,
            message=message,
            severity=severity,
            suggestion=suggestion,
            line_number=self.line_number
        )
        self.issues.append(issue)

    def add_strength(self, description: str):
        """Add a quality strength to the report."""
        self.strengths.append(description)

    def calculate_score(self):
        """Calculate quality score based on issues and strengths."""
        base_score = 100.0

        # Deduct points for issues
        for issue in self.issues:
            if issue.severity == "error":
                base_score -= 20
            elif issue.severity == "warning":
                base_score -= 10
            elif issue.severity == "info":
                base_score -= 5

        # Add points for strengths (capped at original base score)
        strength_bonus = len(self.strengths) * 5
        final_score = min(base_score + strength_bonus, 100.0)

        self.quality_score = max(final_score, 0.0)


@dataclass
class ProjectQualityReport:
    """Overall quality report for a project."""
    project_path: str
    member_reports: List[QualityReport] = field(default_factory=list)
    overall_score: float = 0.0
    summary_stats: Dict[str, int] = field(default_factory=dict)

    def calculate_overall_score(self):
        """Calculate overall project quality score."""
        if not self.member_reports:
            self.overall_score = 100.0
            return

        total_score = sum(report.quality_score for report in self.member_reports)
        self.overall_score = total_score / len(self.member_reports)

        # Calculate summary statistics
        self.summary_stats = {
            'total_members': len(self.member_reports),
            'high_quality': len([r for r in self.member_reports if r.quality_score >= 80]),
            'medium_quality': len([r for r in self.member_reports if 60 <= r.quality_score < 80]),
            'low_quality': len([r for r in self.member_reports if r.quality_score < 60]),
            'total_issues': sum(len(r.issues) for r in self.member_reports),
            'error_count': sum(len([i for i in r.issues if i.severity == "error"]) for r in self.member_reports),
            'warning_count': sum(len([i for i in r.issues if i.severity == "warning"]) for r in self.member_reports)
        }


class DocumentationQualityChecker:
    """Checker for documentation quality and standards compliance."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the quality checker.

        Args:
            config: Configuration dictionary for quality rules
        """
        self.config = config or {}

        # Default configuration
        self.min_docstring_length = self.config.get('min_docstring_length', 20)
        self.max_docstring_length = self.config.get('max_docstring_length', 2000)
        self.require_examples = self.config.get('require_examples', False)
        self.check_grammar = self.config.get('check_grammar', True)
        self.check_spelling = self.config.get('check_spelling', False)
        self.style_guide = self.config.get('style_guide', 'pep257')

        # Grammar and style patterns
        self._load_quality_patterns()

    def _load_quality_patterns(self):
        """Load patterns for detecting quality issues."""
        # Common grammar issues
        self.grammar_patterns = [
            (r'\b(it\'s|its)\b(?!\s+(own|way|time))', "Potential 'its' vs 'it's' confusion"),
            (r'\bthere\s+is\s+\d+', "Consider 'there are' for plural"),
            (r'\band\s+and\b', "Redundant 'and'"),
            (r'\bthe\s+the\b', "Redundant 'the'"),
            (r'\ba\s+[aeiouAEIOU]', "Should use 'an' before vowel sounds"),
        ]

        # Style patterns for different guides
        if self.style_guide == 'pep257':
            self.style_patterns = [
                (r'^[a-z]', "Docstring should start with capital letter", "warning"),
                (r'[^.!?]$', "Docstring should end with punctuation", "warning"),
                (r'^\s*""".*"""$', "Single-line docstring should be on one line", "info"),
            ]
        elif self.style_guide == 'google':
            self.style_patterns = [
                (r'^[a-z]', "Docstring should start with capital letter", "warning"),
                (r'Args:\s*$', "Args section should not be empty", "warning"),
                (r'Returns:\s*$', "Returns section should not be empty", "warning"),
            ]
        else:
            self.style_patterns = []

        # Code example patterns
        self.code_example_patterns = [
            (r'>>>', "Interactive Python example"),
            (r'^\s*```\s*python', "Python code block"),
            (r'^\s*```', "Generic code block"),
            (r'Example[s]?:', "Example section"),
        ]

        # Common spelling errors (basic set)
        self.common_misspellings = {
            'teh': 'the',
            'adn': 'and',
            'recieve': 'receive',
            'occured': 'occurred',
            'seperate': 'separate',
            'definately': 'definitely',
            'neccessary': 'necessary',
            'accomodate': 'accommodate',
        }

    def check_member_quality(self, member: DocumentationNode) -> QualityReport:
        """Check the quality of documentation for a single member.

        Args:
            member: DocumentationNode to analyze

        Returns:
            QualityReport with issues and score
        """
        report = QualityReport(
            member_name=member.name,
            member_type=member.member_type,
            source_file=member.source_file,
            line_number=member.line_number
        )

        if not member.docstring:
            report.add_issue(
                QualityIssueType.LENGTH_TOO_SHORT,
                "No docstring provided",
                severity="error",
                suggestion="Add a descriptive docstring explaining the purpose and usage"
            )
            report.calculate_score()
            return report

        docstring = member.docstring.strip()

        # Check docstring length
        self._check_length(docstring, report)

        # Check grammar and style
        if self.check_grammar:
            self._check_grammar(docstring, report)

        # Check style compliance
        self._check_style_compliance(docstring, report)

        # Check for examples
        self._check_examples(member, docstring, report)

        # Check spelling
        if self.check_spelling:
            self._check_spelling(docstring, report)

        # Check type information consistency
        self._check_type_consistency(member, report)

        # Check for unclear descriptions
        self._check_clarity(docstring, report)

        # Add strengths
        self._identify_strengths(member, docstring, report)

        report.calculate_score()
        return report

    def check_project_quality(self, modules: List[DocumentationNode]) -> ProjectQualityReport:
        """Check quality for an entire project.

        Args:
            modules: List of module DocumentationNode objects

        Returns:
            ProjectQualityReport with overall assessment
        """
        project_report = ProjectQualityReport(project_path="")

        for module in modules:
            # Check module itself
            if module.docstring:
                module_report = self.check_member_quality(module)
                project_report.member_reports.append(module_report)

            # Check all members recursively
            self._check_members_recursive(module, project_report.member_reports)

        project_report.calculate_overall_score()
        return project_report

    def _check_members_recursive(self, node: DocumentationNode, reports: List[QualityReport]):
        """Recursively check all members of a node."""
        for child in node.children:
            if child.docstring:  # Only check members with docstrings
                report = self.check_member_quality(child)
                reports.append(report)

            # Recursively check children
            if child.children:
                self._check_members_recursive(child, reports)

    def _check_length(self, docstring: str, report: QualityReport):
        """Check docstring length."""
        length = len(docstring)

        if length < self.min_docstring_length:
            report.add_issue(
                QualityIssueType.LENGTH_TOO_SHORT,
                f"Docstring is too short ({length} chars, minimum {self.min_docstring_length})",
                severity="warning",
                suggestion="Expand with more detailed description, parameters, and examples"
            )
        elif length > self.max_docstring_length:
            report.add_issue(
                QualityIssueType.LENGTH_TOO_LONG,
                f"Docstring is very long ({length} chars, maximum {self.max_docstring_length})",
                severity="info",
                suggestion="Consider breaking into smaller sections or moving details to separate documentation"
            )
        else:
            report.add_strength("Appropriate docstring length")

    def _check_grammar(self, docstring: str, report: QualityReport):
        """Check for common grammar issues."""
        for pattern, message in self.grammar_patterns:
            matches = re.finditer(pattern, docstring, re.IGNORECASE)
            for match in matches:
                report.add_issue(
                    QualityIssueType.POOR_GRAMMAR,
                    f"Potential grammar issue: {message}",
                    severity="info",
                    suggestion=f"Check text near: '{match.group()}'"
                )

    def _check_style_compliance(self, docstring: str, report: QualityReport):
        """Check compliance with style guide."""
        lines = docstring.split('\n')
        first_line = lines[0] if lines else ""

        for pattern, message, severity in self.style_patterns:
            if re.search(pattern, first_line):
                report.add_issue(
                    QualityIssueType.INCONSISTENT_STYLE,
                    message,
                    severity=severity,
                    suggestion=f"Follow {self.style_guide} style guide"
                )

    def _check_examples(self, member: DocumentationNode, docstring: str, report: QualityReport):
        """Check for and validate examples."""
        has_examples = any(re.search(pattern, docstring, re.MULTILINE)
                          for pattern, _ in self.code_example_patterns)

        if self.require_examples and not has_examples:
            if member.member_type in [MemberType.FUNCTION, MemberType.METHOD]:
                report.add_issue(
                    QualityIssueType.MISSING_EXAMPLES,
                    "Function/method should include usage examples",
                    severity="warning",
                    suggestion="Add examples showing typical usage"
                )
        elif has_examples:
            report.add_strength("Includes code examples")

            # Check for broken examples (basic validation)
            self._validate_code_examples(docstring, report)

    def _validate_code_examples(self, docstring: str, report: QualityReport):
        """Validate code examples in docstring."""
        # Extract Python examples (>>> format)
        example_pattern = r'>>>\s+(.+?)(?=\n(?!\.\.\.)|$)'
        examples = re.findall(example_pattern, docstring, re.MULTILINE | re.DOTALL)

        for example in examples:
            try:
                # Basic syntax check
                compile(example.strip(), '<docstring>', 'eval')
            except SyntaxError:
                try:
                    # Try as statement
                    compile(example.strip(), '<docstring>', 'exec')
                except SyntaxError:
                    report.add_issue(
                        QualityIssueType.BROKEN_CODE_EXAMPLE,
                        f"Code example has syntax error: {example[:50]}...",
                        severity="warning",
                        suggestion="Fix the syntax in the code example"
                    )

    def _check_spelling(self, docstring: str, report: QualityReport):
        """Check for common spelling errors."""
        words = re.findall(r'\b[a-zA-Z]+\b', docstring.lower())

        for word in words:
            if word in self.common_misspellings:
                correct = self.common_misspellings[word]
                report.add_issue(
                    QualityIssueType.SPELLING_ERROR,
                    f"Possible misspelling: '{word}'",
                    severity="info",
                    suggestion=f"Did you mean '{correct}'?"
                )

    def _check_type_consistency(self, member: DocumentationNode, report: QualityReport):
        """Check consistency between type annotations and documentation."""
        if member.member_type not in [MemberType.FUNCTION, MemberType.METHOD]:
            return

        # Check parameter type consistency
        if member.parameters:
            documented_params = set()
            if member.metadata and 'parameters' in member.metadata:
                documented_params = set(member.metadata['parameters'].keys())

            annotated_params = {param.name for param in member.parameters if param.annotation}

            # Parameters with annotations but no documentation
            missing_docs = annotated_params - documented_params
            if missing_docs:
                report.add_issue(
                    QualityIssueType.MISSING_TYPE_INFO,
                    f"Parameters have type annotations but no docs: {', '.join(missing_docs)}",
                    severity="warning",
                    suggestion="Document all annotated parameters"
                )

        # Check return type consistency
        if member.return_annotation and member.return_annotation != 'None':
            if not member.return_description and not (member.metadata and member.metadata.get('returns')):
                report.add_issue(
                    QualityIssueType.MISSING_TYPE_INFO,
                    "Function has return type annotation but no return documentation",
                    severity="warning",
                    suggestion="Document the return value"
                )

    def _check_clarity(self, docstring: str, report: QualityReport):
        """Check for unclear or vague descriptions."""
        vague_phrases = [
            'does stuff', 'handles things', 'processes data', 'works with',
            'manages', 'deals with', 'takes care of'
        ]

        docstring_lower = docstring.lower()
        for phrase in vague_phrases:
            if phrase in docstring_lower:
                report.add_issue(
                    QualityIssueType.UNCLEAR_DESCRIPTION,
                    f"Vague description contains: '{phrase}'",
                    severity="info",
                    suggestion="Be more specific about what the function/class does"
                )

        # Check for very short first line
        first_line = docstring.split('\n')[0].strip()
        if len(first_line) < 10:
            report.add_issue(
                QualityIssueType.UNCLEAR_DESCRIPTION,
                "First line summary is very short",
                severity="info",
                suggestion="Provide a more descriptive summary"
            )

    def _identify_strengths(self, member: DocumentationNode, docstring: str, report: QualityReport):
        """Identify positive aspects of the documentation."""
        lines = docstring.split('\n')

        # Good summary line
        first_line = lines[0].strip()
        if len(first_line) > 20 and first_line.endswith('.'):
            report.add_strength("Clear, well-formed summary line")

        # Well-structured sections
        if 'Args:' in docstring or 'Parameters:' in docstring:
            report.add_strength("Includes parameter documentation")

        if 'Returns:' in docstring or 'Return:' in docstring:
            report.add_strength("Documents return value")

        if 'Raises:' in docstring:
            report.add_strength("Documents exceptions")

        if 'Examples:' in docstring or 'Example:' in docstring:
            report.add_strength("Provides usage examples")

        # Good length and detail
        if len(docstring) > 100:
            report.add_strength("Detailed documentation")

        # Type annotations
        if member.parameters and any(p.annotation for p in member.parameters):
            report.add_strength("Uses type annotations")

        if member.return_annotation:
            report.add_strength("Has return type annotation")

    def generate_quality_report(self, project_report: ProjectQualityReport,
                              output_format: str = 'text') -> str:
        """Generate a formatted quality report.

        Args:
            project_report: ProjectQualityReport to format
            output_format: Output format ('text', 'json', 'html')

        Returns:
            Formatted quality report
        """
        if output_format == 'json':
            return self._generate_json_quality_report(project_report)
        elif output_format == 'html':
            return self._generate_html_quality_report(project_report)
        else:
            return self._generate_text_quality_report(project_report)

    def _generate_text_quality_report(self, project_report: ProjectQualityReport) -> str:
        """Generate text format quality report."""
        lines = []

        lines.append("Documentation Quality Report")
        lines.append("=" * 40)
        lines.append(f"Overall Score: {project_report.overall_score:.1f}/100")
        lines.append("")

        # Summary statistics
        stats = project_report.summary_stats
        lines.append("Summary:")
        lines.append(f"  Total Members: {stats['total_members']}")
        lines.append(f"  High Quality (≥80): {stats['high_quality']}")
        lines.append(f"  Medium Quality (60-79): {stats['medium_quality']}")
        lines.append(f"  Low Quality (<60): {stats['low_quality']}")
        lines.append(f"  Total Issues: {stats['total_issues']}")
        lines.append(f"  Errors: {stats['error_count']}")
        lines.append(f"  Warnings: {stats['warning_count']}")
        lines.append("")

        # Top issues by type
        all_issues = []
        for report in project_report.member_reports:
            all_issues.extend(report.issues)

        issue_counts = {}
        for issue in all_issues:
            issue_type = issue.issue_type.value
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

        if issue_counts:
            lines.append("Most Common Issues:")
            sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
            for issue_type, count in sorted_issues[:5]:
                lines.append(f"  {issue_type.replace('_', ' ').title()}: {count}")
            lines.append("")

        # Member details (worst first)
        lines.append("Member Quality Details:")
        lines.append("-" * 30)

        sorted_reports = sorted(project_report.member_reports,
                              key=lambda x: x.quality_score)

        for report in sorted_reports[:10]:  # Show worst 10
            lines.append(f"{report.member_name} ({report.member_type.value}): {report.quality_score:.1f}/100")

            if report.issues:
                for issue in report.issues[:3]:  # Show first 3 issues
                    lines.append(f"  - {issue.message}")

            if report.strengths:
                lines.append(f"  Strengths: {', '.join(report.strengths[:2])}")

            lines.append("")

        return '\n'.join(lines)

    def _generate_json_quality_report(self, project_report: ProjectQualityReport) -> str:
        """Generate JSON format quality report."""
        import json

        data = {
            'overall_score': project_report.overall_score,
            'summary_stats': project_report.summary_stats,
            'members': []
        }

        for report in project_report.member_reports:
            member_data = {
                'name': report.member_name,
                'type': report.member_type.value,
                'quality_score': report.quality_score,
                'source_file': report.source_file,
                'line_number': report.line_number,
                'issues': [
                    {
                        'type': issue.issue_type.value,
                        'message': issue.message,
                        'severity': issue.severity,
                        'suggestion': issue.suggestion
                    }
                    for issue in report.issues
                ],
                'strengths': report.strengths
            }
            data['members'].append(member_data)

        return json.dumps(data, indent=2)

    def _generate_html_quality_report(self, project_report: ProjectQualityReport) -> str:
        """Generate HTML format quality report."""
        html_parts = [
            '<!DOCTYPE html>',
            '<html><head><title>Documentation Quality Report</title>',
            '<style>',
            'body { font-family: Arial, sans-serif; margin: 40px; }',
            '.score { font-size: 24px; font-weight: bold; margin: 20px 0; }',
            '.high-quality { color: #4caf50; }',
            '.medium-quality { color: #ff9800; }',
            '.low-quality { color: #f44336; }',
            '.member { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }',
            '.issues { margin: 10px 0; }',
            '.issue { margin: 5px 0; padding: 5px; background: #fff3cd; border-left: 3px solid #ffc107; }',
            '.error { background: #f8d7da; border-left-color: #dc3545; }',
            '.strengths { margin: 10px 0; color: #28a745; }',
            '</style>',
            '</head><body>',
            '<h1>Documentation Quality Report</h1>',
            f'<div class="score">Overall Score: {project_report.overall_score:.1f}/100</div>'
        ]

        # Summary stats
        stats = project_report.summary_stats
        html_parts.extend([
            '<h2>Summary</h2>',
            '<ul>',
            f'<li>Total Members: {stats["total_members"]}</li>',
            f'<li class="high-quality">High Quality (≥80): {stats["high_quality"]}</li>',
            f'<li class="medium-quality">Medium Quality (60-79): {stats["medium_quality"]}</li>',
            f'<li class="low_quality">Low Quality (&lt;60): {stats["low_quality"]}</li>',
            f'<li>Total Issues: {stats["total_issues"]}</li>',
            '</ul>'
        ])

        # Member details
        html_parts.append('<h2>Member Details</h2>')

        for report in sorted(project_report.member_reports, key=lambda x: x.quality_score):
            score_class = ('high-quality' if report.quality_score >= 80 else
                          'medium-quality' if report.quality_score >= 60 else 'low-quality')

            html_parts.extend([
                '<div class="member">',
                f'<h3>{report.member_name} <span class="{score_class}">({report.quality_score:.1f}/100)</span></h3>',
                f'<p><em>{report.member_type.value}</em></p>'
            ])

            if report.issues:
                html_parts.append('<div class="issues"><h4>Issues:</h4>')
                for issue in report.issues:
                    issue_class = 'error' if issue.severity == 'error' else 'issue'
                    html_parts.append(f'<div class="{issue_class}">{issue.message}</div>')
                html_parts.append('</div>')

            if report.strengths:
                html_parts.extend([
                    '<div class="strengths"><h4>Strengths:</h4>',
                    '<ul>'
                ])
                for strength in report.strengths:
                    html_parts.append(f'<li>{strength}</li>')
                html_parts.extend(['</ul>', '</div>'])

            html_parts.append('</div>')

        html_parts.extend(['</body>', '</html>'])
        return '\n'.join(html_parts)