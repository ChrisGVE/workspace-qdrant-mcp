"""Template engine for generating documentation from parsed code information."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from jinja2 import Environment, FileSystemLoader, select_autoescape, Template, TemplateNotFound
import yaml

from .ast_parser import DocumentationNode, MemberType


class DocumentationTemplateEngine:
    """Template engine for generating documentation from parsed code."""

    def __init__(self, template_dir: Union[str, Path], config: Optional[Dict[str, Any]] = None):
        """Initialize the template engine.

        Args:
            template_dir: Directory containing Jinja2 templates
            config: Configuration dictionary for template rendering
        """
        self.template_dir = Path(template_dir)
        self.config = config or {}

        # Set up Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )

        # Add custom filters
        self._register_filters()

    def _register_filters(self):
        """Register custom Jinja2 filters."""
        self.env.filters['format_type_annotation'] = self._format_type_annotation
        self.env.filters['format_signature'] = self._format_signature
        self.env.filters['format_docstring'] = self._format_docstring
        self.env.filters['format_member_type'] = self._format_member_type
        self.env.filters['is_public'] = lambda x: not x.is_private if hasattr(x, 'is_private') else True
        self.env.filters['sort_by_name'] = lambda items: sorted(items, key=lambda x: x.name if hasattr(x, 'name') else str(x))
        self.env.filters['group_by_type'] = self._group_by_type

    def render_api_documentation(self, modules: List[DocumentationNode],
                                output_format: str = 'html') -> str:
        """Render API documentation for a list of modules.

        Args:
            modules: List of module DocumentationNode objects
            output_format: Output format ('html', 'markdown', 'json')

        Returns:
            Rendered documentation as a string
        """
        template_name = f'api_template.{output_format}.jinja2'

        try:
            template = self.env.get_template(template_name)
        except TemplateNotFound:
            # Fallback to base template
            template_name = 'api_template.jinja2'
            template = self.env.get_template(template_name)

        context = {
            'modules': modules,
            'config': self.config,
            'project': self.config.get('project', {}),
            'output_format': output_format
        }

        return template.render(**context)

    def render_module_documentation(self, module: DocumentationNode,
                                   output_format: str = 'html') -> str:
        """Render documentation for a single module.

        Args:
            module: Module DocumentationNode
            output_format: Output format ('html', 'markdown', 'json')

        Returns:
            Rendered module documentation
        """
        template_name = f'module_template.{output_format}.jinja2'

        try:
            template = self.env.get_template(template_name)
        except TemplateNotFound:
            template_name = 'module_template.jinja2'
            template = self.env.get_template(template_name)

        # Group children by type for better organization
        classes = [c for c in module.children if c.member_type == MemberType.CLASS]
        functions = [c for c in module.children if c.member_type == MemberType.FUNCTION]
        constants = [c for c in module.children if c.member_type == MemberType.CONSTANT]
        attributes = [c for c in module.children if c.member_type == MemberType.ATTRIBUTE]

        context = {
            'module': module,
            'classes': classes,
            'functions': functions,
            'constants': constants,
            'attributes': attributes,
            'config': self.config,
            'project': self.config.get('project', {}),
            'output_format': output_format
        }

        return template.render(**context)

    def render_class_documentation(self, class_node: DocumentationNode,
                                  output_format: str = 'html') -> str:
        """Render documentation for a single class.

        Args:
            class_node: Class DocumentationNode
            output_format: Output format ('html', 'markdown', 'json')

        Returns:
            Rendered class documentation
        """
        template_name = f'class_template.{output_format}.jinja2'

        try:
            template = self.env.get_template(template_name)
        except TemplateNotFound:
            template_name = 'class_template.jinja2'
            template = self.env.get_template(template_name)

        # Group methods by type
        methods = [c for c in class_node.children if c.member_type == MemberType.METHOD]
        properties = [c for c in class_node.children if c.member_type == MemberType.PROPERTY]
        attributes = [c for c in class_node.children if c.member_type == MemberType.ATTRIBUTE]

        context = {
            'class': class_node,
            'methods': methods,
            'properties': properties,
            'attributes': attributes,
            'config': self.config,
            'project': self.config.get('project', {}),
            'output_format': output_format
        }

        return template.render(**context)

    def render_user_guide(self, guide_data: Dict[str, Any],
                         output_format: str = 'html') -> str:
        """Render user guide documentation.

        Args:
            guide_data: Dictionary containing guide information
            output_format: Output format ('html', 'markdown', 'json')

        Returns:
            Rendered user guide
        """
        template_name = f'guide_template.{output_format}.jinja2'

        try:
            template = self.env.get_template(template_name)
        except TemplateNotFound:
            template_name = 'guide_template.jinja2'
            template = self.env.get_template(template_name)

        context = {
            'guide': guide_data,
            'config': self.config,
            'project': self.config.get('project', {}),
            'output_format': output_format
        }

        return template.render(**context)

    def render_custom_template(self, template_name: str,
                              context: Dict[str, Any]) -> str:
        """Render a custom template with provided context.

        Args:
            template_name: Name of the template file
            context: Context dictionary for rendering

        Returns:
            Rendered template content
        """
        template = self.env.get_template(template_name)

        # Merge with default context
        full_context = {
            'config': self.config,
            'project': self.config.get('project', {}),
            **context
        }

        return template.render(**full_context)

    def create_template_from_string(self, template_string: str) -> Template:
        """Create a template from a string.

        Args:
            template_string: Template content as string

        Returns:
            Compiled Jinja2 template
        """
        return self.env.from_string(template_string)

    def _format_type_annotation(self, annotation: Optional[str]) -> str:
        """Format type annotation for display."""
        if not annotation:
            return ""

        # Basic formatting - could be enhanced with syntax highlighting
        return annotation.replace('typing.', '').replace('__builtin__.', '')

    def _format_signature(self, signature: Optional[str]) -> str:
        """Format function signature for display."""
        if not signature:
            return ""

        # Break long signatures into multiple lines
        if len(signature) > 80:
            # Simple line breaking at commas
            parts = signature.split(', ')
            if len(parts) > 1:
                formatted = parts[0]
                for part in parts[1:]:
                    formatted += ',\n    ' + part
                return formatted

        return signature

    def _format_docstring(self, docstring: Optional[str]) -> str:
        """Format docstring for display."""
        if not docstring:
            return ""

        # Basic formatting - convert to HTML/markdown as needed
        lines = docstring.strip().split('\n')

        # Remove common leading whitespace
        if len(lines) > 1:
            # Find minimum leading whitespace (excluding first line)
            min_indent = float('inf')
            for line in lines[1:]:
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    min_indent = min(min_indent, indent)

            if min_indent != float('inf'):
                lines = [lines[0]] + [line[min_indent:] if len(line) > min_indent else line for line in lines[1:]]

        return '\n'.join(lines)

    def _format_member_type(self, member_type: MemberType) -> str:
        """Format member type for display."""
        return member_type.value.replace('_', ' ').title()

    def _group_by_type(self, items: List[DocumentationNode]) -> Dict[str, List[DocumentationNode]]:
        """Group items by their member type."""
        groups = {}
        for item in items:
            type_name = item.member_type.value
            if type_name not in groups:
                groups[type_name] = []
            groups[type_name].append(item)
        return groups

    def list_templates(self) -> List[str]:
        """List available templates in the template directory.

        Returns:
            List of template filenames
        """
        templates = []
        for file_path in self.template_dir.rglob('*.jinja2'):
            relative_path = file_path.relative_to(self.template_dir)
            templates.append(str(relative_path))
        return sorted(templates)

    def validate_template(self, template_name: str) -> bool:
        """Validate that a template exists and can be loaded.

        Args:
            template_name: Name of the template to validate

        Returns:
            True if template is valid, False otherwise
        """
        try:
            self.env.get_template(template_name)
            return True
        except TemplateNotFound:
            return False

    def get_template_dependencies(self, template_name: str) -> List[str]:
        """Get list of templates that this template depends on (includes/extends).

        Args:
            template_name: Name of the template to analyze

        Returns:
            List of dependency template names
        """
        try:
            template = self.env.get_template(template_name)
            # This is a simplified approach - a full implementation would
            # parse the template AST to find all includes/extends
            return []
        except TemplateNotFound:
            return []


def create_default_templates(template_dir: Union[str, Path]):
    """Create default documentation templates.

    Args:
        template_dir: Directory where templates should be created
    """
    template_dir = Path(template_dir)
    template_dir.mkdir(parents=True, exist_ok=True)

    # Base template
    base_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}{{ project.name or "Documentation" }}{% endblock %}</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { border-bottom: 1px solid #eee; margin-bottom: 30px; padding-bottom: 20px; }
        .module { margin-bottom: 40px; }
        .class { margin-bottom: 30px; border-left: 3px solid #007acc; padding-left: 20px; }
        .function { margin-bottom: 20px; }
        .signature { background: #f8f8f8; padding: 10px; border-radius: 4px; font-family: monospace; }
        .docstring { margin: 15px 0; line-height: 1.6; }
        .parameters { margin: 10px 0; }
        .parameter { margin: 5px 0; }
        .private { opacity: 0.7; }
        pre { background: #f8f8f8; padding: 15px; border-radius: 4px; overflow-x: auto; }
        code { background: #f0f0f0; padding: 2px 4px; border-radius: 2px; font-family: monospace; }
    </style>
    {% block extra_head %}{% endblock %}
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{% block page_title %}{{ project.name or "Documentation" }}{% endblock %}</h1>
            <p>{% block description %}{{ project.description }}{% endblock %}</p>
        </div>
        <div class="content">
            {% block content %}{% endblock %}
        </div>
    </div>
</body>
</html>'''

    (template_dir / 'base.jinja2').write_text(base_template)

    # API template
    api_template = '''{% extends "base.jinja2" %}

{% block title %}API Documentation - {{ project.name }}{% endblock %}
{% block page_title %}API Documentation{% endblock %}

{% block content %}
<div class="api-overview">
    <h2>Modules</h2>
    {% for module in modules %}
    <div class="module">
        <h3>{{ module.name }}</h3>
        {% if module.docstring %}
        <div class="docstring">{{ module.docstring | format_docstring }}</div>
        {% endif %}

        {% set classes = module.children | selectattr("member_type.value", "equalto", "class") | list %}
        {% if classes %}
        <h4>Classes</h4>
        <ul>
        {% for class in classes %}
            <li>
                <strong>{{ class.name }}</strong>
                {% if class.docstring %}
                - {{ class.docstring.split('\n')[0] }}
                {% endif %}
            </li>
        {% endfor %}
        </ul>
        {% endif %}

        {% set functions = module.children | selectattr("member_type.value", "equalto", "function") | list %}
        {% if functions %}
        <h4>Functions</h4>
        <ul>
        {% for func in functions %}
            <li>
                <code>{{ func.signature or func.name }}</code>
                {% if func.docstring %}
                - {{ func.docstring.split('\n')[0] }}
                {% endif %}
            </li>
        {% endfor %}
        </ul>
        {% endif %}
    </div>
    {% endfor %}
</div>
{% endblock %}'''

    (template_dir / 'api_template.jinja2').write_text(api_template)

    # Module template
    module_template = '''{% extends "base.jinja2" %}

{% block title %}{{ module.name }} - {{ project.name }}{% endblock %}
{% block page_title %}Module: {{ module.name }}{% endblock %}

{% block content %}
<div class="module-doc">
    {% if module.docstring %}
    <div class="module-description">
        <div class="docstring">{{ module.docstring | format_docstring }}</div>
    </div>
    {% endif %}

    {% if classes %}
    <section class="classes">
        <h2>Classes</h2>
        {% for class in classes | sort_by_name %}
        <div class="class">
            <h3>{{ class.name }}</h3>
            {% if class.signature %}
            <div class="signature">{{ class.signature | format_signature }}</div>
            {% endif %}
            {% if class.docstring %}
            <div class="docstring">{{ class.docstring | format_docstring }}</div>
            {% endif %}

            {% set methods = class.children | selectattr("member_type.value", "equalto", "method") | list %}
            {% if methods %}
            <h4>Methods</h4>
            {% for method in methods | sort_by_name %}
            <div class="function">
                <h5>{{ method.name }}</h5>
                {% if method.signature %}
                <div class="signature">{{ method.signature | format_signature }}</div>
                {% endif %}
                {% if method.docstring %}
                <div class="docstring">{{ method.docstring | format_docstring }}</div>
                {% endif %}
            </div>
            {% endfor %}
            {% endif %}
        </div>
        {% endfor %}
    </section>
    {% endif %}

    {% if functions %}
    <section class="functions">
        <h2>Functions</h2>
        {% for func in functions | sort_by_name %}
        <div class="function">
            <h3>{{ func.name }}</h3>
            {% if func.signature %}
            <div class="signature">{{ func.signature | format_signature }}</div>
            {% endif %}
            {% if func.docstring %}
            <div class="docstring">{{ func.docstring | format_docstring }}</div>
            {% endif %}
        </div>
        {% endfor %}
    </section>
    {% endif %}

    {% if constants %}
    <section class="constants">
        <h2>Constants</h2>
        {% for const in constants | sort_by_name %}
        <div class="constant">
            <h3>{{ const.name }}</h3>
            {% if const.docstring %}
            <div class="docstring">{{ const.docstring | format_docstring }}</div>
            {% endif %}
        </div>
        {% endfor %}
    </section>
    {% endif %}
</div>
{% endblock %}'''

    (template_dir / 'module_template.jinja2').write_text(module_template)