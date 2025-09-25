"""Consistency checker for documentation standards and style compliance."""

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any, Callable
from collections import defaultdict, Counter
import yaml
import json

logger = logging.getLogger(__name__)


class ConsistencyLevel(Enum):
    """Severity levels for consistency violations."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ConsistencyRule:
    """Defines a consistency rule to check."""

    name: str
    description: str
    level: ConsistencyLevel
    pattern: Optional[str] = None
    checker_function: Optional[Callable] = None
    expected_value: Optional[str] = None
    applies_to: List[str] = field(default_factory=lambda: ['*.md', '*.rst'])
    enabled: bool = True


@dataclass
class ConsistencyViolation:
    """Represents a consistency rule violation."""

    rule_name: str
    file_path: Path
    line_number: int
    column: int
    message: str
    level: ConsistencyLevel
    context: str = ""
    suggested_fix: Optional[str] = None
    actual_value: Optional[str] = None
    expected_value: Optional[str] = None

    def __post_init__(self):
        """Initialize computed fields."""
        if isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)


@dataclass
class ConsistencyReport:
    """Report of consistency checking results."""

    total_files_checked: int = 0
    total_violations: int = 0
    violations_by_level: Dict[ConsistencyLevel, int] = field(default_factory=dict)
    violations_by_rule: Dict[str, int] = field(default_factory=dict)
    violations: List[ConsistencyViolation] = field(default_factory=list)
    compliance_score: float = 0.0

    def __post_init__(self):
        """Initialize computed fields."""
        if not self.violations_by_level:
            for level in ConsistencyLevel:
                self.violations_by_level[level] = 0


class ConsistencyChecker:
    """Checks documentation consistency according to style guides and standards."""

    def __init__(self,
                 root_path: Union[str, Path],
                 config_file: Optional[Union[str, Path]] = None,
                 custom_rules: Optional[List[ConsistencyRule]] = None):
        """Initialize consistency checker.

        Args:
            root_path: Root directory for documentation
            config_file: Path to configuration file with custom rules
            custom_rules: Additional custom rules to apply
        """
        self.root_path = Path(root_path)
        self.rules: Dict[str, ConsistencyRule] = {}

        # Load default rules
        self._load_default_rules()

        # Load custom rules from config file
        if config_file:
            self._load_config_file(config_file)

        # Add custom rules
        if custom_rules:
            for rule in custom_rules:
                self.rules[rule.name] = rule

        # Statistics tracking
        self._file_patterns: Set[str] = set()
        self._terminology: Dict[str, int] = defaultdict(int)

    def _load_default_rules(self):
        """Load default consistency rules."""

        # Heading style rules
        self.rules['heading_capitalization'] = ConsistencyRule(
            name='heading_capitalization',
            description='Check heading capitalization consistency',
            level=ConsistencyLevel.WARNING,
            pattern=r'^#{1,6}\s+(.+)$',
            checker_function=self._check_heading_capitalization
        )

        self.rules['heading_spacing'] = ConsistencyRule(
            name='heading_spacing',
            description='Check consistent spacing around headings',
            level=ConsistencyLevel.INFO,
            pattern=r'^#{1,6}',
            checker_function=self._check_heading_spacing
        )

        # Code block consistency
        self.rules['code_block_language'] = ConsistencyRule(
            name='code_block_language',
            description='Check code blocks have language specified',
            level=ConsistencyLevel.WARNING,
            pattern=r'^```(\w*)',
            checker_function=self._check_code_block_language
        )

        # Link consistency
        self.rules['link_format'] = ConsistencyRule(
            name='link_format',
            description='Check consistent link formatting',
            level=ConsistencyLevel.INFO,
            pattern=r'\[([^\]]+)\]\(([^)]+)\)',
            checker_function=self._check_link_format
        )

        # Terminology consistency
        self.rules['terminology_consistency'] = ConsistencyRule(
            name='terminology_consistency',
            description='Check consistent use of terminology',
            level=ConsistencyLevel.ERROR,
            checker_function=self._check_terminology_consistency
        )

        # List formatting
        self.rules['list_formatting'] = ConsistencyRule(
            name='list_formatting',
            description='Check consistent list item formatting',
            level=ConsistencyLevel.INFO,
            pattern=r'^\s*[-*+]\s',
            checker_function=self._check_list_formatting
        )

        # Table formatting
        self.rules['table_formatting'] = ConsistencyRule(
            name='table_formatting',
            description='Check table formatting consistency',
            level=ConsistencyLevel.WARNING,
            pattern=r'^\s*\|',
            checker_function=self._check_table_formatting
        )

        # Line length
        self.rules['line_length'] = ConsistencyRule(
            name='line_length',
            description='Check line length consistency',
            level=ConsistencyLevel.INFO,
            expected_value='120',
            checker_function=self._check_line_length
        )

        # Punctuation in headings
        self.rules['heading_punctuation'] = ConsistencyRule(
            name='heading_punctuation',
            description='Check headings end without periods',
            level=ConsistencyLevel.WARNING,
            pattern=r'^#{1,6}\s+(.+)\.\s*$',
            checker_function=self._check_heading_punctuation
        )

        # Consistent quote style
        self.rules['quote_style'] = ConsistencyRule(
            name='quote_style',
            description='Check consistent quotation mark style',
            level=ConsistencyLevel.INFO,
            checker_function=self._check_quote_style
        )

    def _load_config_file(self, config_file: Path):
        """Load rules from configuration file."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.suffix == '.yaml' or config_file.suffix == '.yml':
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)

            # Load rules from config
            for rule_data in config.get('rules', []):
                rule = ConsistencyRule(
                    name=rule_data['name'],
                    description=rule_data['description'],
                    level=ConsistencyLevel(rule_data['level']),
                    pattern=rule_data.get('pattern'),
                    expected_value=rule_data.get('expected_value'),
                    applies_to=rule_data.get('applies_to', ['*.md', '*.rst']),
                    enabled=rule_data.get('enabled', True)
                )
                self.rules[rule.name] = rule

        except Exception as e:
            logger.error(f"Failed to load config file {config_file}: {e}")

    def check_consistency(self, file_patterns: List[str] = None) -> ConsistencyReport:
        """Check consistency across documentation files.

        Args:
            file_patterns: Patterns to match files for checking

        Returns:
            ConsistencyReport with detailed violations
        """
        if file_patterns is None:
            file_patterns = ['**/*.md', '**/*.rst', '**/*.txt']

        logger.info(f"Starting consistency check in {self.root_path}")

        report = ConsistencyReport()

        # First pass: collect terminology and patterns
        self._collect_statistics(file_patterns)

        # Second pass: check consistency
        for pattern in file_patterns:
            for file_path in self.root_path.glob(pattern):
                if not file_path.is_file():
                    continue

                try:
                    violations = self._check_file_consistency(file_path)
                    report.violations.extend(violations)
                    report.total_files_checked += 1

                except Exception as e:
                    logger.error(f"Error checking {file_path}: {e}")

        # Calculate statistics
        report.total_violations = len(report.violations)

        for violation in report.violations:
            report.violations_by_level[violation.level] += 1
            report.violations_by_rule[violation.rule_name] += 1

        # Calculate compliance score (100% - error percentage)
        if report.total_files_checked > 0:
            error_weight = {
                ConsistencyLevel.CRITICAL: 4,
                ConsistencyLevel.ERROR: 3,
                ConsistencyLevel.WARNING: 2,
                ConsistencyLevel.INFO: 1
            }

            total_weight = sum(
                report.violations_by_level.get(level, 0) * weight
                for level, weight in error_weight.items()
            )

            # Normalize to 0-100 scale
            max_possible_weight = report.total_files_checked * 10  # Arbitrary max
            report.compliance_score = max(0, 100 - (total_weight / max_possible_weight * 100))

        logger.info(f"Consistency check complete: {report.total_violations} violations found "
                   f"across {report.total_files_checked} files "
                   f"(compliance score: {report.compliance_score:.1f}%)")

        return report

    def _collect_statistics(self, file_patterns: List[str]):
        """Collect statistics for consistency analysis."""
        logger.debug("Collecting documentation statistics")

        for pattern in file_patterns:
            for file_path in self.root_path.glob(pattern):
                if not file_path.is_file():
                    continue

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Collect terminology
                    words = re.findall(r'\b\w{3,}\b', content.lower())
                    for word in words:
                        self._terminology[word] += 1

                except Exception as e:
                    logger.debug(f"Could not read {file_path}: {e}")

    def _check_file_consistency(self, file_path: Path) -> List[ConsistencyViolation]:
        """Check consistency rules for a single file."""
        violations = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Could not read {file_path}: {e}")
            return violations

        lines = content.split('\n')

        # Apply each enabled rule
        for rule in self.rules.values():
            if not rule.enabled:
                continue

            # Check if rule applies to this file type
            file_extension = f"*{file_path.suffix}"
            if not any(fnmatch.fnmatch(file_extension, pattern) for pattern in rule.applies_to):
                continue

            if rule.checker_function:
                try:
                    rule_violations = rule.checker_function(rule, file_path, lines, content)
                    violations.extend(rule_violations)
                except Exception as e:
                    logger.error(f"Error applying rule {rule.name} to {file_path}: {e}")

        return violations

    def _check_heading_capitalization(self, rule: ConsistencyRule,
                                    file_path: Path, lines: List[str],
                                    content: str) -> List[ConsistencyViolation]:
        """Check heading capitalization consistency."""
        violations = []
        heading_styles = Counter()

        for line_num, line in enumerate(lines, 1):
            match = re.match(rule.pattern, line)
            if match:
                heading_text = match.group(1)

                # Analyze capitalization style
                if heading_text.istitle():
                    style = 'title_case'
                elif heading_text.isupper():
                    style = 'upper_case'
                elif heading_text.islower():
                    style = 'lower_case'
                else:
                    style = 'mixed_case'

                heading_styles[style] += 1

        # Find most common style
        if heading_styles:
            most_common_style = heading_styles.most_common(1)[0][0]

            # Check each heading against the most common style
            for line_num, line in enumerate(lines, 1):
                match = re.match(rule.pattern, line)
                if match:
                    heading_text = match.group(1)

                    current_style = 'mixed_case'
                    if heading_text.istitle():
                        current_style = 'title_case'
                    elif heading_text.isupper():
                        current_style = 'upper_case'
                    elif heading_text.islower():
                        current_style = 'lower_case'

                    if current_style != most_common_style:
                        violations.append(ConsistencyViolation(
                            rule_name=rule.name,
                            file_path=file_path,
                            line_number=line_num,
                            column=0,
                            message=f"Heading capitalization inconsistent: found {current_style}, expected {most_common_style}",
                            level=rule.level,
                            context=line.strip(),
                            actual_value=current_style,
                            expected_value=most_common_style
                        ))

        return violations

    def _check_heading_spacing(self, rule: ConsistencyRule,
                             file_path: Path, lines: List[str],
                             content: str) -> List[ConsistencyViolation]:
        """Check consistent spacing around headings."""
        violations = []

        for line_num, line in enumerate(lines, 1):
            if re.match(rule.pattern, line):
                # Check spacing before heading (except first line)
                if line_num > 1:
                    prev_line = lines[line_num - 2]
                    if prev_line.strip() and not prev_line.startswith('#'):
                        violations.append(ConsistencyViolation(
                            rule_name=rule.name,
                            file_path=file_path,
                            line_number=line_num,
                            column=0,
                            message="Heading should have blank line before it",
                            level=rule.level,
                            context=line.strip(),
                            suggested_fix="Add blank line before heading"
                        ))

        return violations

    def _check_code_block_language(self, rule: ConsistencyRule,
                                  file_path: Path, lines: List[str],
                                  content: str) -> List[ConsistencyViolation]:
        """Check code blocks have language specified."""
        violations = []

        for line_num, line in enumerate(lines, 1):
            match = re.match(rule.pattern, line)
            if match and line.startswith('```'):
                language = match.group(1)
                if not language:
                    violations.append(ConsistencyViolation(
                        rule_name=rule.name,
                        file_path=file_path,
                        line_number=line_num,
                        column=0,
                        message="Code block missing language specification",
                        level=rule.level,
                        context=line.strip(),
                        suggested_fix="Add language after ```"
                    ))

        return violations

    def _check_link_format(self, rule: ConsistencyRule,
                          file_path: Path, lines: List[str],
                          content: str) -> List[ConsistencyViolation]:
        """Check consistent link formatting."""
        violations = []
        link_formats = Counter()

        # Collect all link formats
        for line_num, line in enumerate(lines, 1):
            # Markdown links
            md_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', line)
            # Reference links
            ref_links = re.findall(r'\[([^\]]+)\]\[([^\]]*)\]', line)

            if md_links:
                link_formats['markdown'] += len(md_links)
            if ref_links:
                link_formats['reference'] += len(ref_links)

        # If mixed formats, suggest consistency
        if len(link_formats) > 1:
            preferred_format = link_formats.most_common(1)[0][0]

            for line_num, line in enumerate(lines, 1):
                if preferred_format == 'markdown' and re.search(r'\[([^\]]+)\]\[([^\]]*)\]', line):
                    violations.append(ConsistencyViolation(
                        rule_name=rule.name,
                        file_path=file_path,
                        line_number=line_num,
                        column=0,
                        message=f"Inconsistent link format: prefer {preferred_format} style",
                        level=rule.level,
                        context=line.strip()
                    ))

        return violations

    def _check_terminology_consistency(self, rule: ConsistencyRule,
                                     file_path: Path, lines: List[str],
                                     content: str) -> List[ConsistencyViolation]:
        """Check consistent terminology usage."""
        violations = []

        # Define common technical term variations
        term_variants = {
            'api': ['api', 'API', 'Api'],
            'json': ['json', 'JSON', 'Json'],
            'http': ['http', 'HTTP', 'Http'],
            'url': ['url', 'URL', 'Url'],
            'database': ['database', 'Database', 'db', 'DB'],
            'javascript': ['javascript', 'JavaScript', 'JS', 'js'],
            'python': ['python', 'Python'],
        }

        for base_term, variants in term_variants.items():
            found_variants = set()

            for line_num, line in enumerate(lines, 1):
                for variant in variants:
                    if re.search(r'\b' + re.escape(variant) + r'\b', line):
                        found_variants.add(variant)

            # If multiple variants found, suggest consistency
            if len(found_variants) > 1:
                # Prefer the most common variant
                variant_counts = Counter()
                for variant in found_variants:
                    variant_counts[variant] = self._terminology.get(variant.lower(), 0)

                preferred = variant_counts.most_common(1)[0][0]

                for line_num, line in enumerate(lines, 1):
                    for variant in found_variants:
                        if variant != preferred and re.search(r'\b' + re.escape(variant) + r'\b', line):
                            violations.append(ConsistencyViolation(
                                rule_name=rule.name,
                                file_path=file_path,
                                line_number=line_num,
                                column=line.find(variant),
                                message=f"Inconsistent terminology: '{variant}' should be '{preferred}'",
                                level=rule.level,
                                context=line.strip(),
                                actual_value=variant,
                                expected_value=preferred
                            ))

        return violations

    def _check_list_formatting(self, rule: ConsistencyRule,
                              file_path: Path, lines: List[str],
                              content: str) -> List[ConsistencyViolation]:
        """Check consistent list formatting."""
        violations = []
        list_markers = Counter()

        # Collect list markers
        for line in lines:
            match = re.match(rule.pattern, line)
            if match:
                marker = match.group(0).strip().split()[0]  # Get the marker (-, *, +)
                list_markers[marker] += 1

        # Check for consistency if multiple markers found
        if len(list_markers) > 1:
            preferred_marker = list_markers.most_common(1)[0][0]

            for line_num, line in enumerate(lines, 1):
                match = re.match(rule.pattern, line)
                if match:
                    marker = match.group(0).strip().split()[0]
                    if marker != preferred_marker:
                        violations.append(ConsistencyViolation(
                            rule_name=rule.name,
                            file_path=file_path,
                            line_number=line_num,
                            column=0,
                            message=f"Inconsistent list marker: '{marker}' should be '{preferred_marker}'",
                            level=rule.level,
                            context=line.strip(),
                            actual_value=marker,
                            expected_value=preferred_marker
                        ))

        return violations

    def _check_table_formatting(self, rule: ConsistencyRule,
                               file_path: Path, lines: List[str],
                               content: str) -> List[ConsistencyViolation]:
        """Check table formatting consistency."""
        violations = []

        in_table = False
        table_start = 0

        for line_num, line in enumerate(lines, 1):
            if re.match(rule.pattern, line):
                if not in_table:
                    in_table = True
                    table_start = line_num

                # Check column alignment consistency within table
                if '|' in line:
                    cells = [cell.strip() for cell in line.split('|')]
                    # Check for consistent spacing
                    for i, cell in enumerate(cells):
                        if cell and not cell.replace('-', '').strip():  # Header separator
                            continue
                        # Additional table formatting checks can be added here

            elif in_table and line.strip() and not re.match(rule.pattern, line):
                in_table = False
                # End of table - could add table-level checks here

        return violations

    def _check_line_length(self, rule: ConsistencyRule,
                          file_path: Path, lines: List[str],
                          content: str) -> List[ConsistencyViolation]:
        """Check line length consistency."""
        violations = []
        max_length = int(rule.expected_value) if rule.expected_value else 120

        for line_num, line in enumerate(lines, 1):
            # Skip code blocks and tables
            if line.strip().startswith('```') or line.strip().startswith('|'):
                continue

            if len(line) > max_length:
                violations.append(ConsistencyViolation(
                    rule_name=rule.name,
                    file_path=file_path,
                    line_number=line_num,
                    column=max_length,
                    message=f"Line too long: {len(line)} characters (max: {max_length})",
                    level=rule.level,
                    context=line[:50] + "..." if len(line) > 50 else line.strip(),
                    actual_value=str(len(line)),
                    expected_value=str(max_length)
                ))

        return violations

    def _check_heading_punctuation(self, rule: ConsistencyRule,
                                  file_path: Path, lines: List[str],
                                  content: str) -> List[ConsistencyViolation]:
        """Check headings don't end with periods."""
        violations = []

        for line_num, line in enumerate(lines, 1):
            if re.match(rule.pattern, line):
                violations.append(ConsistencyViolation(
                    rule_name=rule.name,
                    file_path=file_path,
                    line_number=line_num,
                    column=len(line) - 1,
                    message="Heading should not end with period",
                    level=rule.level,
                    context=line.strip(),
                    suggested_fix="Remove period from heading"
                ))

        return violations

    def _check_quote_style(self, rule: ConsistencyRule,
                          file_path: Path, lines: List[str],
                          content: str) -> List[ConsistencyViolation]:
        """Check consistent quotation mark style."""
        violations = []
        quote_styles = Counter()

        # Count different quote styles
        single_quotes = len(re.findall(r"'[^']*'", content))
        double_quotes = len(re.findall(r'"[^"]*"', content))

        if single_quotes > 0:
            quote_styles['single'] = single_quotes
        if double_quotes > 0:
            quote_styles['double'] = double_quotes

        # If mixed styles, suggest consistency
        if len(quote_styles) > 1:
            preferred = quote_styles.most_common(1)[0][0]
            non_preferred = 'single' if preferred == 'double' else 'double'
            pattern = r"'[^']*'" if non_preferred == 'single' else r'"[^"]*"'

            for line_num, line in enumerate(lines, 1):
                if re.search(pattern, line):
                    violations.append(ConsistencyViolation(
                        rule_name=rule.name,
                        file_path=file_path,
                        line_number=line_num,
                        column=0,
                        message=f"Inconsistent quote style: prefer {preferred} quotes",
                        level=rule.level,
                        context=line.strip(),
                        actual_value=non_preferred,
                        expected_value=preferred
                    ))

        return violations

    def export_report(self, report: ConsistencyReport, output_path: Path) -> bool:
        """Export consistency report to file."""
        try:
            export_data = {
                'summary': {
                    'total_files_checked': report.total_files_checked,
                    'total_violations': report.total_violations,
                    'compliance_score': report.compliance_score,
                    'violations_by_level': {
                        level.value: count
                        for level, count in report.violations_by_level.items()
                    },
                    'violations_by_rule': report.violations_by_rule
                },
                'violations': [
                    {
                        'rule_name': v.rule_name,
                        'file_path': str(v.file_path),
                        'line_number': v.line_number,
                        'column': v.column,
                        'level': v.level.value,
                        'message': v.message,
                        'context': v.context,
                        'suggested_fix': v.suggested_fix,
                        'actual_value': v.actual_value,
                        'expected_value': v.expected_value
                    }
                    for v in report.violations
                ],
                'rules_applied': {
                    name: {
                        'description': rule.description,
                        'level': rule.level.value,
                        'enabled': rule.enabled
                    }
                    for name, rule in self.rules.items()
                }
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Consistency report exported to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export report: {e}")
            return False

# Import fnmatch for file pattern matching
import fnmatch