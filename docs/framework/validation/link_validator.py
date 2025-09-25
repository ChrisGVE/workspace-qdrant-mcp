"""Link validator for checking documentation references and URLs."""

import re
import asyncio
import aiohttp
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urljoin, urlparse
import time

try:
    from ..generators.ast_parser import DocumentationNode
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from generators.ast_parser import DocumentationNode


class LinkType(Enum):
    """Types of links that can be validated."""
    HTTP_URL = "http_url"
    HTTPS_URL = "https_url"
    LOCAL_FILE = "local_file"
    INTERNAL_REFERENCE = "internal_reference"
    EMAIL = "email"
    FTP_URL = "ftp_url"
    UNKNOWN = "unknown"


class LinkStatus(Enum):
    """Status of link validation."""
    VALID = "valid"
    BROKEN = "broken"
    REDIRECT = "redirect"
    TIMEOUT = "timeout"
    UNKNOWN_ERROR = "unknown_error"
    NOT_FOUND = "not_found"
    FILE_NOT_EXISTS = "file_not_exists"
    PERMISSION_DENIED = "permission_denied"


@dataclass
class LinkInfo:
    """Information about a link found in documentation."""
    url: str
    link_type: LinkType
    context: str = ""  # Surrounding text
    source_file: Optional[str] = None
    line_number: Optional[int] = None
    member_name: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of validating a single link."""
    link: LinkInfo
    status: LinkStatus
    status_code: Optional[int] = None
    redirect_url: Optional[str] = None
    error_message: Optional[str] = None
    response_time: Optional[float] = None
    final_url: Optional[str] = None


@dataclass
class ValidationReport:
    """Report of link validation for a project."""
    project_path: str
    results: List[ValidationResult] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    validation_time: float = 0.0

    def calculate_summary(self):
        """Calculate summary statistics."""
        self.summary = {
            'total_links': len(self.results),
            'valid_links': len([r for r in self.results if r.status == LinkStatus.VALID]),
            'broken_links': len([r for r in self.results if r.status == LinkStatus.BROKEN]),
            'redirected_links': len([r for r in self.results if r.status == LinkStatus.REDIRECT]),
            'timeout_links': len([r for r in self.results if r.status == LinkStatus.TIMEOUT]),
            'error_links': len([r for r in self.results if r.status == LinkStatus.UNKNOWN_ERROR]),
        }

        # Calculate success rate
        if self.summary['total_links'] > 0:
            success_count = (self.summary['valid_links'] +
                           self.summary['redirected_links'])
            self.summary['success_rate'] = (success_count / self.summary['total_links']) * 100
        else:
            self.summary['success_rate'] = 100.0


class LinkValidator:
    """Validator for checking links in documentation."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the link validator.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Configuration settings
        self.timeout = self.config.get('timeout', 10.0)
        self.max_concurrent = self.config.get('max_concurrent', 10)
        self.check_external_links = self.config.get('check_external_links', True)
        self.check_local_files = self.config.get('check_local_files', True)
        self.follow_redirects = self.config.get('follow_redirects', True)
        self.user_agent = self.config.get('user_agent',
            'Documentation Link Validator 1.0 (Python/aiohttp)')

        # Rate limiting
        self.delay_between_requests = self.config.get('delay_between_requests', 0.1)
        self.max_retries = self.config.get('max_retries', 2)

        # Exclusions
        self.excluded_domains = set(self.config.get('excluded_domains', []))
        self.excluded_patterns = self.config.get('excluded_patterns', [])

        # Compile regex patterns
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for link detection."""
        # URL patterns
        self.url_pattern = re.compile(
            r'https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*)?(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?',
            re.IGNORECASE
        )

        # Markdown link patterns
        self.markdown_link_pattern = re.compile(
            r'\[([^\]]*)\]\(([^)]+)\)',
            re.IGNORECASE
        )

        # RST link patterns
        self.rst_link_pattern = re.compile(
            r'`([^`<]+)\s*<([^>]+)>`_',
            re.IGNORECASE
        )

        # Email pattern
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            re.IGNORECASE
        )

        # Local file references
        self.local_file_pattern = re.compile(
            r'(?:\.{1,2}/|/)[a-zA-Z0-9_/.-]+\.(?:py|md|txt|rst|html|pdf|doc|docx)',
            re.IGNORECASE
        )

    def extract_links_from_docstring(self, docstring: str,
                                   source_file: str = None,
                                   member_name: str = None) -> List[LinkInfo]:
        """Extract all links from a docstring.

        Args:
            docstring: The docstring to analyze
            source_file: Source file path for context
            member_name: Member name for context

        Returns:
            List of LinkInfo objects
        """
        if not docstring:
            return []

        links = []

        # Extract different types of links
        links.extend(self._extract_url_links(docstring, source_file, member_name))
        links.extend(self._extract_markdown_links(docstring, source_file, member_name))
        links.extend(self._extract_rst_links(docstring, source_file, member_name))
        links.extend(self._extract_email_links(docstring, source_file, member_name))
        links.extend(self._extract_local_file_links(docstring, source_file, member_name))

        # Remove duplicates while preserving order
        seen = set()
        unique_links = []
        for link in links:
            key = (link.url, link.link_type)
            if key not in seen:
                seen.add(key)
                unique_links.append(link)

        return unique_links

    def _extract_url_links(self, text: str, source_file: str = None,
                          member_name: str = None) -> List[LinkInfo]:
        """Extract HTTP/HTTPS URLs from text."""
        links = []
        for match in self.url_pattern.finditer(text):
            url = match.group(0)
            context = self._get_context(text, match.start(), match.end())

            link_type = LinkType.HTTPS_URL if url.startswith('https') else LinkType.HTTP_URL

            links.append(LinkInfo(
                url=url,
                link_type=link_type,
                context=context,
                source_file=source_file,
                member_name=member_name
            ))

        return links

    def _extract_markdown_links(self, text: str, source_file: str = None,
                               member_name: str = None) -> List[LinkInfo]:
        """Extract Markdown format links."""
        links = []
        for match in self.markdown_link_pattern.finditer(text):
            link_text = match.group(1)
            url = match.group(2)
            context = f"[{link_text}]({url})"

            link_type = self._determine_link_type(url)

            links.append(LinkInfo(
                url=url,
                link_type=link_type,
                context=context,
                source_file=source_file,
                member_name=member_name
            ))

        return links

    def _extract_rst_links(self, text: str, source_file: str = None,
                          member_name: str = None) -> List[LinkInfo]:
        """Extract reStructuredText format links."""
        links = []
        for match in self.rst_link_pattern.finditer(text):
            link_text = match.group(1)
            url = match.group(2)
            context = f"`{link_text} <{url}>`_"

            link_type = self._determine_link_type(url)

            links.append(LinkInfo(
                url=url,
                link_type=link_type,
                context=context,
                source_file=source_file,
                member_name=member_name
            ))

        return links

    def _extract_email_links(self, text: str, source_file: str = None,
                           member_name: str = None) -> List[LinkInfo]:
        """Extract email addresses."""
        links = []
        for match in self.email_pattern.finditer(text):
            email = match.group(0)
            context = self._get_context(text, match.start(), match.end())

            links.append(LinkInfo(
                url=f"mailto:{email}",
                link_type=LinkType.EMAIL,
                context=context,
                source_file=source_file,
                member_name=member_name
            ))

        return links

    def _extract_local_file_links(self, text: str, source_file: str = None,
                                 member_name: str = None) -> List[LinkInfo]:
        """Extract local file references."""
        links = []
        for match in self.local_file_pattern.finditer(text):
            file_path = match.group(0)
            context = self._get_context(text, match.start(), match.end())

            links.append(LinkInfo(
                url=file_path,
                link_type=LinkType.LOCAL_FILE,
                context=context,
                source_file=source_file,
                member_name=member_name
            ))

        return links

    def _get_context(self, text: str, start: int, end: int, context_length: int = 50) -> str:
        """Get context around a match."""
        context_start = max(0, start - context_length)
        context_end = min(len(text), end + context_length)

        return text[context_start:context_end].replace('\n', ' ').strip()

    def _determine_link_type(self, url: str) -> LinkType:
        """Determine the type of a URL."""
        url_lower = url.lower()

        if url_lower.startswith('https://'):
            return LinkType.HTTPS_URL
        elif url_lower.startswith('http://'):
            return LinkType.HTTP_URL
        elif url_lower.startswith('ftp://'):
            return LinkType.FTP_URL
        elif url_lower.startswith('mailto:'):
            return LinkType.EMAIL
        elif url.startswith('/') or url.startswith('./') or url.startswith('../'):
            return LinkType.LOCAL_FILE
        elif '#' in url or url.startswith('#'):
            return LinkType.INTERNAL_REFERENCE
        else:
            return LinkType.UNKNOWN

    def extract_links_from_nodes(self, nodes: List[DocumentationNode]) -> List[LinkInfo]:
        """Extract links from a list of documentation nodes.

        Args:
            nodes: List of DocumentationNode objects

        Returns:
            List of all found links
        """
        all_links = []

        for node in nodes:
            # Extract from node itself
            node_links = self.extract_links_from_docstring(
                node.docstring,
                node.source_file,
                node.name
            )
            all_links.extend(node_links)

            # Recursively extract from children
            if node.children:
                child_links = self.extract_links_from_nodes(node.children)
                all_links.extend(child_links)

        return all_links

    async def validate_links(self, links: List[LinkInfo],
                           project_root: str = None) -> ValidationReport:
        """Validate a list of links.

        Args:
            links: List of LinkInfo objects to validate
            project_root: Root directory for resolving relative paths

        Returns:
            ValidationReport with results
        """
        start_time = time.time()

        report = ValidationReport(project_path=project_root or "")

        # Filter out excluded links
        filtered_links = self._filter_excluded_links(links)

        # Group links by type
        http_links = [link for link in filtered_links
                     if link.link_type in [LinkType.HTTP_URL, LinkType.HTTPS_URL]]
        local_links = [link for link in filtered_links
                      if link.link_type == LinkType.LOCAL_FILE]
        other_links = [link for link in filtered_links
                      if link.link_type not in [LinkType.HTTP_URL, LinkType.HTTPS_URL, LinkType.LOCAL_FILE]]

        # Validate different types of links
        if self.check_external_links and http_links:
            http_results = await self._validate_http_links(http_links)
            report.results.extend(http_results)

        if self.check_local_files and local_links:
            local_results = self._validate_local_links(local_links, project_root)
            report.results.extend(local_results)

        # Handle other link types (currently just mark as valid)
        for link in other_links:
            result = ValidationResult(
                link=link,
                status=LinkStatus.VALID,
                error_message="Email and internal reference links are not validated"
            )
            report.results.append(result)

        report.validation_time = time.time() - start_time
        report.calculate_summary()

        return report

    def _filter_excluded_links(self, links: List[LinkInfo]) -> List[LinkInfo]:
        """Filter out excluded links based on configuration."""
        filtered = []

        for link in links:
            # Check excluded domains
            if self.excluded_domains:
                parsed = urlparse(link.url)
                if parsed.netloc in self.excluded_domains:
                    continue

            # Check excluded patterns
            exclude = False
            for pattern in self.excluded_patterns:
                if re.search(pattern, link.url):
                    exclude = True
                    break

            if not exclude:
                filtered.append(link)

        return filtered

    async def _validate_http_links(self, links: List[LinkInfo]) -> List[ValidationResult]:
        """Validate HTTP/HTTPS links."""
        results = []

        # Create semaphore for limiting concurrent requests
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers={'User-Agent': self.user_agent}
        ) as session:

            tasks = [
                self._validate_single_http_link(session, semaphore, link)
                for link in links
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Create error result
                    error_result = ValidationResult(
                        link=links[i],
                        status=LinkStatus.UNKNOWN_ERROR,
                        error_message=str(result)
                    )
                    valid_results.append(error_result)
                else:
                    valid_results.append(result)

            return valid_results

    async def _validate_single_http_link(self, session: aiohttp.ClientSession,
                                       semaphore: asyncio.Semaphore,
                                       link: LinkInfo) -> ValidationResult:
        """Validate a single HTTP link."""
        async with semaphore:
            start_time = time.time()

            try:
                # Add delay for rate limiting
                if self.delay_between_requests > 0:
                    await asyncio.sleep(self.delay_between_requests)

                async with session.get(
                    link.url,
                    allow_redirects=self.follow_redirects
                ) as response:
                    response_time = time.time() - start_time

                    if response.status == 200:
                        status = LinkStatus.VALID
                    elif 300 <= response.status < 400:
                        status = LinkStatus.REDIRECT
                    elif response.status == 404:
                        status = LinkStatus.NOT_FOUND
                    else:
                        status = LinkStatus.BROKEN

                    return ValidationResult(
                        link=link,
                        status=status,
                        status_code=response.status,
                        final_url=str(response.url),
                        response_time=response_time
                    )

            except asyncio.TimeoutError:
                return ValidationResult(
                    link=link,
                    status=LinkStatus.TIMEOUT,
                    error_message=f"Request timed out after {self.timeout}s"
                )
            except aiohttp.ClientError as e:
                return ValidationResult(
                    link=link,
                    status=LinkStatus.UNKNOWN_ERROR,
                    error_message=str(e)
                )
            except Exception as e:
                return ValidationResult(
                    link=link,
                    status=LinkStatus.UNKNOWN_ERROR,
                    error_message=f"Unexpected error: {str(e)}"
                )

    def _validate_local_links(self, links: List[LinkInfo],
                            project_root: str = None) -> List[ValidationResult]:
        """Validate local file links."""
        results = []

        for link in links:
            try:
                # Resolve relative paths
                if project_root and not Path(link.url).is_absolute():
                    file_path = Path(project_root) / link.url
                else:
                    file_path = Path(link.url)

                if file_path.exists():
                    status = LinkStatus.VALID
                    error_message = None
                else:
                    status = LinkStatus.FILE_NOT_EXISTS
                    error_message = f"File does not exist: {file_path}"

            except PermissionError:
                status = LinkStatus.PERMISSION_DENIED
                error_message = "Permission denied accessing file"
            except Exception as e:
                status = LinkStatus.UNKNOWN_ERROR
                error_message = str(e)

            result = ValidationResult(
                link=link,
                status=status,
                error_message=error_message
            )
            results.append(result)

        return results

    def generate_report(self, report: ValidationReport,
                       output_format: str = 'text') -> str:
        """Generate a formatted validation report.

        Args:
            report: ValidationReport to format
            output_format: Output format ('text', 'json', 'html')

        Returns:
            Formatted report as string
        """
        if output_format == 'json':
            return self._generate_json_report(report)
        elif output_format == 'html':
            return self._generate_html_report(report)
        else:
            return self._generate_text_report(report)

    def _generate_text_report(self, report: ValidationReport) -> str:
        """Generate text format link validation report."""
        lines = []

        lines.append("Link Validation Report")
        lines.append("=" * 30)
        lines.append(f"Validation Time: {report.validation_time:.2f}s")
        lines.append("")

        # Summary
        summary = report.summary
        lines.append("Summary:")
        lines.append(f"  Total Links: {summary['total_links']}")
        lines.append(f"  Valid: {summary['valid_links']}")
        lines.append(f"  Broken: {summary['broken_links']}")
        lines.append(f"  Redirected: {summary['redirected_links']}")
        lines.append(f"  Timeout: {summary['timeout_links']}")
        lines.append(f"  Errors: {summary['error_links']}")
        lines.append(f"  Success Rate: {summary['success_rate']:.1f}%")
        lines.append("")

        # Group results by status
        broken_results = [r for r in report.results if r.status == LinkStatus.BROKEN]
        timeout_results = [r for r in report.results if r.status == LinkStatus.TIMEOUT]
        error_results = [r for r in report.results if r.status == LinkStatus.UNKNOWN_ERROR]

        if broken_results:
            lines.append("Broken Links:")
            lines.append("-" * 15)
            for result in broken_results:
                lines.append(f"  {result.link.url}")
                lines.append(f"    Status: {result.status_code}")
                lines.append(f"    Context: {result.link.context}")
                if result.link.source_file:
                    lines.append(f"    File: {result.link.source_file}")
                lines.append("")

        if timeout_results:
            lines.append("Timeout Links:")
            lines.append("-" * 15)
            for result in timeout_results:
                lines.append(f"  {result.link.url}")
                lines.append(f"    Error: {result.error_message}")
                if result.link.source_file:
                    lines.append(f"    File: {result.link.source_file}")
                lines.append("")

        if error_results:
            lines.append("Error Links:")
            lines.append("-" * 12)
            for result in error_results:
                lines.append(f"  {result.link.url}")
                lines.append(f"    Error: {result.error_message}")
                if result.link.source_file:
                    lines.append(f"    File: {result.link.source_file}")
                lines.append("")

        return '\n'.join(lines)

    def _generate_json_report(self, report: ValidationReport) -> str:
        """Generate JSON format link validation report."""
        import json

        data = {
            'validation_time': report.validation_time,
            'summary': report.summary,
            'results': []
        }

        for result in report.results:
            result_data = {
                'url': result.link.url,
                'link_type': result.link.link_type.value,
                'status': result.status.value,
                'status_code': result.status_code,
                'response_time': result.response_time,
                'error_message': result.error_message,
                'context': result.link.context,
                'source_file': result.link.source_file,
                'member_name': result.link.member_name
            }
            data['results'].append(result_data)

        return json.dumps(data, indent=2)

    def _generate_html_report(self, report: ValidationReport) -> str:
        """Generate HTML format link validation report."""
        html_parts = [
            '<!DOCTYPE html>',
            '<html><head><title>Link Validation Report</title>',
            '<style>',
            'body { font-family: Arial, sans-serif; margin: 40px; }',
            '.summary { background: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }',
            '.valid { color: #28a745; }',
            '.broken { color: #dc3545; }',
            '.redirect { color: #ffc107; }',
            '.timeout { color: #6c757d; }',
            '.result { margin: 15px 0; padding: 10px; border: 1px solid #dee2e6; border-radius: 3px; }',
            '.url { font-family: monospace; font-weight: bold; }',
            '.context { font-style: italic; color: #6c757d; }',
            '</style>',
            '</head><body>',
            '<h1>Link Validation Report</h1>',
            f'<p>Validation completed in {report.validation_time:.2f} seconds</p>'
        ]

        # Summary
        summary = report.summary
        html_parts.extend([
            '<div class="summary">',
            '<h2>Summary</h2>',
            f'<p>Total Links: {summary["total_links"]}</p>',
            f'<p class="valid">Valid: {summary["valid_links"]}</p>',
            f'<p class="broken">Broken: {summary["broken_links"]}</p>',
            f'<p class="redirect">Redirected: {summary["redirected_links"]}</p>',
            f'<p class="timeout">Timeout: {summary["timeout_links"]}</p>',
            f'<p>Success Rate: {summary["success_rate"]:.1f}%</p>',
            '</div>'
        ])

        # Results
        html_parts.append('<h2>Results</h2>')

        for result in report.results:
            status_class = result.status.value.replace('_', '-')

            html_parts.extend([
                f'<div class="result {status_class}">',
                f'<div class="url">{result.link.url}</div>',
                f'<div>Status: <span class="{status_class}">{result.status.value}</span></div>'
            ])

            if result.status_code:
                html_parts.append(f'<div>Status Code: {result.status_code}</div>')

            if result.error_message:
                html_parts.append(f'<div>Error: {result.error_message}</div>')

            if result.link.context:
                html_parts.append(f'<div class="context">Context: {result.link.context}</div>')

            if result.link.source_file:
                html_parts.append(f'<div>File: {result.link.source_file}</div>')

            html_parts.append('</div>')

        html_parts.extend(['</body>', '</html>'])
        return '\n'.join(html_parts)

    def get_broken_links(self, report: ValidationReport) -> List[ValidationResult]:
        """Get all broken links from a validation report.

        Args:
            report: ValidationReport to analyze

        Returns:
            List of ValidationResult objects for broken links
        """
        return [r for r in report.results
                if r.status in [LinkStatus.BROKEN, LinkStatus.NOT_FOUND,
                              LinkStatus.FILE_NOT_EXISTS, LinkStatus.TIMEOUT]]

    def get_success_rate(self, report: ValidationReport) -> float:
        """Calculate success rate from validation report.

        Args:
            report: ValidationReport to analyze

        Returns:
            Success rate as percentage (0-100)
        """
        return report.summary.get('success_rate', 0.0)