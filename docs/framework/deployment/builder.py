"""Documentation builder for generating static documentation sites."""

import os
import shutil
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

try:
    from ..generators.ast_parser import PythonASTParser, DocumentationNode
    from ..generators.template_engine import DocumentationTemplateEngine
    from ..validation.coverage_analyzer import DocumentationCoverageAnalyzer
    from ..validation.quality_checker import DocumentationQualityChecker
    from ..validation.link_validator import LinkValidator
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from generators.ast_parser import PythonASTParser, DocumentationNode
    from generators.template_engine import DocumentationTemplateEngine
    from validation.coverage_analyzer import DocumentationCoverageAnalyzer
    from validation.quality_checker import DocumentationQualityChecker
    from validation.link_validator import LinkValidator


class DocumentationBuilder:
    """Builder for generating complete documentation sites."""

    def __init__(self, config_path: str):
        """Initialize the documentation builder.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Initialize components
        self.ast_parser = PythonASTParser(
            include_private=self.config.get('api', {}).get('include_private', False)
        )
        self.template_engine = None
        self.coverage_analyzer = DocumentationCoverageAnalyzer()
        self.quality_checker = DocumentationQualityChecker(
            config=self.config.get('validation', {}).get('quality', {})
        )
        self.link_validator = LinkValidator(
            config=self.config.get('validation', {}).get('links', {})
        )

        # Set up template engine
        template_dir = self.config_path.parent / "templates"
        if template_dir.exists():
            self.template_engine = DocumentationTemplateEngine(
                template_dir, self.config
            )

        # Build output directory
        self.output_dir = Path(self.config.get('output', {}).get('base_dir', 'docs/generated'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            if self.config_path.suffix.lower() in ['.yml', '.yaml']:
                return yaml.safe_load(f)
            else:
                return json.load(f)

    def build_documentation(self, output_formats: Optional[List[str]] = None) -> Dict[str, Any]:
        """Build complete documentation.

        Args:
            output_formats: List of output formats to generate

        Returns:
            Build report with statistics and paths
        """
        output_formats = output_formats or self.config.get('output', {}).get('formats', ['html'])

        build_report = {
            'success': True,
            'modules_processed': 0,
            'formats_generated': [],
            'validation_results': {},
            'output_paths': {},
            'errors': []
        }

        try:
            # Parse source code
            modules = self._parse_source_code()
            build_report['modules_processed'] = len(modules)

            # Generate documentation in requested formats
            for format_type in output_formats:
                try:
                    output_path = self._generate_format(modules, format_type)
                    build_report['formats_generated'].append(format_type)
                    build_report['output_paths'][format_type] = str(output_path)
                except Exception as e:
                    build_report['errors'].append(f"Failed to generate {format_type}: {e}")

            # Run validation
            if self.config.get('validation', {}).get('enabled', True):
                validation_results = self._run_validation(modules)
                build_report['validation_results'] = validation_results

            # Copy static assets
            self._copy_static_assets()

            # Generate index files
            self._generate_index_files(modules)

        except Exception as e:
            build_report['success'] = False
            build_report['errors'].append(f"Build failed: {e}")

        return build_report

    def _parse_source_code(self) -> List[DocumentationNode]:
        """Parse source code to extract documentation."""
        modules = []

        # Parse Python sources
        python_sources = self.config.get('sources', {}).get('python', [])
        for source_dir in python_sources:
            source_path = Path(source_dir)
            if not source_path.exists():
                continue

            try:
                parsed_modules = self.ast_parser.parse_directory(source_path, recursive=True)
                modules.extend(parsed_modules)
            except Exception as e:
                print(f"Warning: Could not parse {source_dir}: {e}")

        return modules

    def _generate_format(self, modules: List[DocumentationNode], format_type: str) -> Path:
        """Generate documentation in specified format.

        Args:
            modules: List of parsed modules
            format_type: Output format ('html', 'markdown', 'json')

        Returns:
            Path to generated output directory
        """
        format_dir = self.output_dir / format_type
        format_dir.mkdir(parents=True, exist_ok=True)

        if format_type == 'html':
            return self._generate_html(modules, format_dir)
        elif format_type == 'markdown':
            return self._generate_markdown(modules, format_dir)
        elif format_type == 'json':
            return self._generate_json(modules, format_dir)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def _generate_html(self, modules: List[DocumentationNode], output_dir: Path) -> Path:
        """Generate HTML documentation."""
        if not self.template_engine:
            raise RuntimeError("Template engine not available")

        # Generate main API documentation
        api_html = self.template_engine.render_api_documentation(modules, 'html')
        (output_dir / 'index.html').write_text(api_html, encoding='utf-8')

        # Generate individual module pages
        for module in modules:
            module_html = self.template_engine.render_module_documentation(module, 'html')
            module_file = output_dir / f"{module.name}.html"
            module_file.write_text(module_html, encoding='utf-8')

            # Generate class pages for classes in the module
            for child in module.children:
                if child.member_type.value == 'class':
                    class_html = self.template_engine.render_class_documentation(child, 'html')
                    class_file = output_dir / f"{module.name}.{child.name}.html"
                    class_file.write_text(class_html, encoding='utf-8')

        return output_dir

    def _generate_markdown(self, modules: List[DocumentationNode], output_dir: Path) -> Path:
        """Generate Markdown documentation."""
        if not self.template_engine:
            # Fallback to basic markdown generation
            return self._generate_basic_markdown(modules, output_dir)

        # Generate with templates
        for module in modules:
            module_md = self.template_engine.render_module_documentation(module, 'markdown')
            module_file = output_dir / f"{module.name}.md"
            module_file.write_text(module_md, encoding='utf-8')

        # Generate README
        readme_content = self._generate_readme(modules)
        (output_dir / 'README.md').write_text(readme_content, encoding='utf-8')

        return output_dir

    def _generate_basic_markdown(self, modules: List[DocumentationNode], output_dir: Path) -> Path:
        """Generate basic markdown without templates."""
        for module in modules:
            md_content = self._module_to_markdown(module)
            module_file = output_dir / f"{module.name}.md"
            module_file.write_text(md_content, encoding='utf-8')

        return output_dir

    def _module_to_markdown(self, module: DocumentationNode) -> str:
        """Convert a module to markdown format."""
        lines = [f"# {module.name}"]

        if module.docstring:
            lines.extend(["", module.docstring, ""])

        # Group children by type
        classes = [c for c in module.children if c.member_type.value == 'class']
        functions = [c for c in module.children if c.member_type.value == 'function']

        if classes:
            lines.extend(["## Classes", ""])
            for cls in classes:
                lines.extend([f"### {cls.name}", ""])
                if cls.docstring:
                    lines.extend([cls.docstring, ""])

                # Class methods
                methods = [c for c in cls.children if c.member_type.value == 'method']
                if methods:
                    lines.extend(["#### Methods", ""])
                    for method in methods:
                        lines.append(f"- **{method.name}**")
                        if method.signature:
                            lines.append(f"  - Signature: `{method.signature}`")
                        if method.docstring:
                            lines.append(f"  - {method.docstring.split(chr(10))[0]}")
                    lines.append("")

        if functions:
            lines.extend(["## Functions", ""])
            for func in functions:
                lines.extend([f"### {func.name}", ""])
                if func.signature:
                    lines.extend([f"```python", func.signature, "```", ""])
                if func.docstring:
                    lines.extend([func.docstring, ""])

        return "\n".join(lines)

    def _generate_json(self, modules: List[DocumentationNode], output_dir: Path) -> Path:
        """Generate JSON documentation."""
        # Convert modules to serializable format
        json_data = {
            'project': self.config.get('project', {}),
            'generated_at': self._get_timestamp(),
            'modules': []
        }

        for module in modules:
            module_data = self._serialize_module(module)
            json_data['modules'].append(module_data)

        # Write main JSON file
        (output_dir / 'documentation.json').write_text(
            json.dumps(json_data, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )

        # Write individual module files
        for module in modules:
            module_data = self._serialize_module(module)
            module_file = output_dir / f"{module.name}.json"
            module_file.write_text(
                json.dumps(module_data, indent=2, ensure_ascii=False),
                encoding='utf-8'
            )

        return output_dir

    def _serialize_module(self, module: DocumentationNode) -> Dict[str, Any]:
        """Serialize a module to dictionary."""
        return {
            'name': module.name,
            'type': module.member_type.value,
            'docstring': module.docstring,
            'signature': module.signature,
            'source_file': module.source_file,
            'line_number': module.line_number,
            'is_private': module.is_private,
            'children': [self._serialize_module(child) for child in module.children],
            'parameters': [
                {
                    'name': param.name,
                    'annotation': param.annotation,
                    'default': param.default,
                    'kind': param.kind,
                    'description': param.description
                }
                for param in module.parameters
            ] if module.parameters else [],
            'return_annotation': module.return_annotation,
            'return_description': module.return_description,
            'examples': module.examples,
            'decorators': module.decorators,
            'metadata': module.metadata
        }

    def _run_validation(self, modules: List[DocumentationNode]) -> Dict[str, Any]:
        """Run validation checks on documentation."""
        validation_results = {}

        # Coverage analysis
        try:
            coverage_reports = []
            for source_dir in self.config.get('sources', {}).get('python', []):
                if Path(source_dir).exists():
                    coverage = self.coverage_analyzer.analyze_directory(source_dir)
                    coverage_reports.append({
                        'directory': source_dir,
                        'overall_score': coverage.overall_stats.coverage_percentage,
                        'total_items': coverage.overall_stats.total_items,
                        'documented_items': coverage.overall_stats.documented_items
                    })

            validation_results['coverage'] = {
                'overall_score': sum(r['overall_score'] for r in coverage_reports) / len(coverage_reports) if coverage_reports else 0,
                'directories': coverage_reports
            }
        except Exception as e:
            validation_results['coverage'] = {'error': str(e)}

        # Quality analysis
        try:
            quality_report = self.quality_checker.check_project_quality(modules)
            validation_results['quality'] = {
                'overall_score': quality_report.overall_score,
                'summary_stats': quality_report.summary_stats,
                'high_quality_members': quality_report.summary_stats.get('high_quality', 0),
                'low_quality_members': quality_report.summary_stats.get('low_quality', 0)
            }
        except Exception as e:
            validation_results['quality'] = {'error': str(e)}

        # Link validation
        try:
            links = self.link_validator.extract_links_from_nodes(modules)
            if links:
                import asyncio
                link_report = asyncio.run(
                    self.link_validator.validate_links(links, str(self.config_path.parent))
                )
                validation_results['links'] = {
                    'total_links': link_report.summary['total_links'],
                    'success_rate': link_report.summary['success_rate'],
                    'broken_links': link_report.summary['broken_links']
                }
            else:
                validation_results['links'] = {'total_links': 0, 'success_rate': 100.0}
        except Exception as e:
            validation_results['links'] = {'error': str(e)}

        return validation_results

    def _copy_static_assets(self):
        """Copy static assets to output directory."""
        static_sources = [
            self.config_path.parent / "static",
            self.config_path.parent / "assets",
        ]

        static_output = self.output_dir / "static"

        for static_source in static_sources:
            if static_source.exists():
                if static_output.exists():
                    shutil.rmtree(static_output)
                shutil.copytree(static_source, static_output, dirs_exist_ok=True)

    def _generate_index_files(self, modules: List[DocumentationNode]):
        """Generate index files for documentation."""
        # Generate sitemap
        sitemap_content = self._generate_sitemap(modules)
        (self.output_dir / 'sitemap.txt').write_text(sitemap_content, encoding='utf-8')

        # Generate search index
        search_index = self._generate_search_index(modules)
        (self.output_dir / 'search-index.json').write_text(
            json.dumps(search_index, ensure_ascii=False),
            encoding='utf-8'
        )

    def _generate_sitemap(self, modules: List[DocumentationNode]) -> str:
        """Generate a sitemap for the documentation."""
        base_url = self.config.get('deployment', {}).get('base_url', '')
        urls = []

        # Add main pages
        urls.extend([
            f"{base_url}/",
            f"{base_url}/api/",
        ])

        # Add module pages
        for module in modules:
            urls.append(f"{base_url}/api/{module.name}.html")

        return '\n'.join(urls)

    def _generate_search_index(self, modules: List[DocumentationNode]) -> Dict[str, Any]:
        """Generate search index for client-side search."""
        index = {
            'modules': [],
            'members': []
        }

        for module in modules:
            # Add module to index
            module_entry = {
                'name': module.name,
                'type': 'module',
                'docstring': module.docstring,
                'url': f"/api/{module.name}.html"
            }
            index['modules'].append(module_entry)

            # Add members to index
            self._add_members_to_index(module, index['members'], f"/api/{module.name}.html")

        return index

    def _add_members_to_index(self, node: DocumentationNode, members_index: List[Dict],
                             base_url: str):
        """Recursively add members to search index."""
        for child in node.children:
            member_entry = {
                'name': child.name,
                'full_name': f"{node.name}.{child.name}",
                'type': child.member_type.value,
                'docstring': child.docstring,
                'signature': child.signature,
                'url': f"{base_url}#{child.name}"
            }
            members_index.append(member_entry)

            # Recursively add children
            if child.children:
                self._add_members_to_index(child, members_index, base_url)

    def _generate_readme(self, modules: List[DocumentationNode]) -> str:
        """Generate README.md content."""
        project_name = self.config.get('project', {}).get('name', 'Documentation')
        project_description = self.config.get('project', {}).get('description', '')

        lines = [
            f"# {project_name}",
            "",
        ]

        if project_description:
            lines.extend([project_description, ""])

        lines.extend([
            "## Modules",
            "",
        ])

        for module in modules:
            lines.append(f"- [{module.name}]({module.name}.md)")
            if module.docstring:
                first_line = module.docstring.split('\n')[0]
                lines.append(f"  - {first_line}")

        lines.extend([
            "",
            f"Generated on {self._get_timestamp()}",
        ])

        return '\n'.join(lines)

    def _get_timestamp(self) -> str:
        """Get current timestamp as string."""
        from datetime import datetime
        return datetime.now().isoformat()

    def clean_output(self):
        """Clean the output directory."""
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

    def get_build_info(self) -> Dict[str, Any]:
        """Get information about the build configuration."""
        return {
            'config_path': str(self.config_path),
            'output_dir': str(self.output_dir),
            'sources': self.config.get('sources', {}),
            'formats': self.config.get('output', {}).get('formats', []),
            'validation_enabled': self.config.get('validation', {}).get('enabled', True)
        }