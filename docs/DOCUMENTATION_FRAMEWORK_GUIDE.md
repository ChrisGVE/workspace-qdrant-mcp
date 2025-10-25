# Documentation Framework Guide

This guide explains how to use the comprehensive documentation framework built for the Workspace Qdrant MCP project.

## Overview

The documentation framework consists of five main components:

1. **AST Parser** - Extracts documentation from Python code using AST analysis
2. **Template Engine** - Generates documentation in multiple formats using Jinja2
3. **Validation System** - Analyzes coverage, quality, and link validity
4. **Interactive Server** - Provides live documentation with executable examples
5. **Deployment System** - Builds and deploys documentation to various platforms

## Quick Start

### 1. Configure Documentation

Create a `docs/config.yaml` file:

```yaml
project:
  name: "Your Project Name"
  description: "Project description"
  version: "1.0.0"
  author: "Your Name"

sources:
  python:
    - "src/python"
  rust:
    - "rust-engine-legacy"

output:
  base_dir: "docs/generated"
  formats:
    - html
    - markdown
    - json

api:
  include_private: false
  show_source_links: true

validation:
  coverage:
    minimum_percentage: 80
    require_examples: true
    require_return_docs: true
    require_param_docs: true
  quality:
    min_docstring_length: 20
    check_grammar: true
  links:
    check_external_links: true
    timeout: 10

server:
  host: "127.0.0.1"
  port: 8080
  auto_reload: true
  enable_sandbox: true

deployment:
  github_pages: true
  base_url: "/your-project/"
```

### 2. Generate Documentation

```python
from docs.framework.deployment.builder import DocumentationBuilder

# Build documentation
builder = DocumentationBuilder("docs/config.yaml")
report = builder.build_documentation()

print(f"Built documentation for {report['modules_processed']} modules")
print(f"Generated formats: {report['formats_generated']}")
```

### 3. Run Interactive Server

```python
from docs.framework.server.app import run_server
import yaml

# Load configuration
with open("docs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Start server
run_server(config, host="127.0.0.1", port=8080)
```

Then visit http://127.0.0.1:8080 to explore your documentation interactively.

## Components Usage

### AST Parser

Extract documentation from Python code:

```python
from docs.framework.generators.ast_parser import PythonASTParser

parser = PythonASTParser(include_private=False)

# Parse single file
module = parser.parse_file("path/to/module.py")

# Parse directory
modules = parser.parse_directory("src/python", recursive=True)

# Access parsed information
print(f"Module: {module.name}")
print(f"Docstring: {module.docstring}")
print(f"Members: {len(module.children)}")

for child in module.children:
    print(f"  {child.name} ({child.member_type.value})")
    if child.parameters:
        for param in child.parameters:
            print(f"    {param.name}: {param.annotation}")
```

### Template Engine

Generate documentation with templates:

```python
from docs.framework.generators.template_engine import DocumentationTemplateEngine

engine = DocumentationTemplateEngine("docs/templates", config)

# Generate API documentation
html_docs = engine.render_api_documentation(modules, "html")
markdown_docs = engine.render_api_documentation(modules, "markdown")

# Generate module-specific documentation
module_html = engine.render_module_documentation(modules[0], "html")

# Use custom templates
custom_output = engine.render_custom_template("my_template.jinja2", {
    "modules": modules,
    "custom_data": "value"
})
```

### Coverage Analysis

Analyze documentation coverage:

```python
from docs.framework.validation.coverage_analyzer import DocumentationCoverageAnalyzer

analyzer = DocumentationCoverageAnalyzer(
    require_examples=True,
    require_return_docs=True,
    require_param_docs=True
)

# Analyze single file
file_coverage = analyzer.analyze_file("src/module.py")
print(f"Coverage: {file_coverage.stats.coverage_percentage:.1f}%")

# Analyze entire project
project_coverage = analyzer.analyze_directory("src/", recursive=True)

# Generate reports
text_report = analyzer.generate_report(project_coverage, "text")
html_report = analyzer.generate_report(project_coverage, "html")
json_report = analyzer.generate_report(project_coverage, "json")

# Find issues
undocumented = analyzer.find_undocumented_members(project_coverage)
meets_threshold = analyzer.meets_threshold(project_coverage, 80.0)
```

### Quality Checking

Assess documentation quality:

```python
from docs.framework.validation.quality_checker import DocumentationQualityChecker

checker = DocumentationQualityChecker(config={
    'min_docstring_length': 20,
    'check_grammar': True,
    'require_examples': True,
    'style_guide': 'pep257'
})

# Check project quality
quality_report = checker.check_project_quality(modules)

print(f"Overall Score: {quality_report.overall_score:.1f}/100")
print(f"High Quality Members: {quality_report.summary_stats['high_quality']}")
print(f"Issues Found: {quality_report.summary_stats['total_issues']}")

# Generate quality report
report_text = checker.generate_quality_report(quality_report, "text")
report_html = checker.generate_quality_report(quality_report, "html")
```

### Link Validation

Validate links in documentation:

```python
import asyncio
from docs.framework.validation.link_validator import LinkValidator

validator = LinkValidator(config={
    'timeout': 10,
    'check_external_links': True,
    'max_concurrent': 5,
    'excluded_domains': ['example.com']
})

# Extract links from documentation
links = validator.extract_links_from_nodes(modules)
print(f"Found {len(links)} links")

# Validate links
async def validate():
    report = await validator.validate_links(links, project_root=".")

    print(f"Success Rate: {report.summary['success_rate']:.1f}%")
    print(f"Broken Links: {report.summary['broken_links']}")

    # Get broken links
    broken = validator.get_broken_links(report)
    for result in broken:
        print(f"  {result.link.url}: {result.error_message}")

asyncio.run(validate())
```

### Code Sandbox

Execute code examples safely:

```python
import asyncio
from docs.framework.server.sandbox import CodeSandbox

sandbox = CodeSandbox(timeout=30, memory_limit=128)

async def test_code():
    # Execute simple code
    result = await sandbox.execute_code("2 + 2")
    print(f"Result: {result['result']}")

    # Execute with context
    result = await sandbox.execute_code(
        "name.upper()",
        context={"name": "hello world"}
    )
    print(f"Output: {result['result']}")

    # Test example validation
    examples = [
        "x = 5",
        "print(x * 2)",
        "import os"  # This should be blocked
    ]

    validation_results = sandbox.validate_examples(examples)
    for result in validation_results:
        print(f"Example {result['index']}: {'✓' if result['valid'] else '✗'}")

asyncio.run(test_code())
```

## Template Customization

### Creating Custom Templates

1. Create template files in `docs/templates/`:

```jinja2
<!-- custom_template.html.jinja2 -->
<!DOCTYPE html>
<html>
<head>
    <title>{{ project.name }} - Custom Documentation</title>
</head>
<body>
    <h1>{{ project.name }}</h1>

    {% for module in modules %}
    <section>
        <h2>{{ module.name }}</h2>
        {% if module.docstring %}
        <p>{{ module.docstring | format_docstring }}</p>
        {% endif %}

        {% for member in module.children | is_public %}
        <div class="member">
            <h3>{{ member.name }}</h3>
            {% if member.signature %}
            <code>{{ member.signature | format_signature }}</code>
            {% endif %}
        </div>
        {% endfor %}
    </section>
    {% endfor %}
</body>
</html>
```

2. Use custom filters in templates:

```jinja2
{{ docstring | format_docstring }}
{{ signature | format_signature }}
{{ annotation | format_type_annotation }}
{{ members | group_by_type }}
{{ items | sort_by_name }}
{{ member | is_public }}
```

### Available Template Variables

- `project`: Project configuration
- `modules`: List of parsed modules
- `config`: Full configuration dictionary
- `output_format`: Current output format

## API Server Endpoints

The interactive server provides these endpoints:

- `GET /` - Main documentation interface
- `GET /api/modules` - List all modules
- `GET /api/modules/{name}` - Get module details
- `GET /api/modules/{name}/coverage` - Get coverage analysis
- `GET /api/modules/{name}/quality` - Get quality analysis
- `POST /api/execute` - Execute code in sandbox
- `POST /api/search` - Search documentation
- `GET /api/examples/{module}/{member}` - Get member examples
- `GET /health` - Health check

### Example API Usage

```javascript
// Search documentation
const searchResponse = await fetch('/api/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        query: 'authentication',
        member_types: ['function', 'class'],
        include_private: false
    })
});

// Execute code
const executeResponse = await fetch('/api/execute', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        code: 'print("Hello, World!")',
        language: 'python',
        context: { 'name': 'World' }
    })
});
```

## Deployment

### GitHub Pages Deployment

```python
from docs.framework.deployment.deployer import DocumentationDeployer
from docs.framework.deployment.builder import DocumentationBuilder

# Build documentation
builder = DocumentationBuilder("docs/config.yaml")
build_report = builder.build_documentation()

# Deploy to GitHub Pages
deployer = DocumentationDeployer(config)
deploy_result = deployer.deploy_to_github_pages(
    Path("docs/generated/html"),
    branch="gh-pages"
)

if deploy_result['success']:
    print(f"Deployed to: {deploy_result['url']}")
else:
    print(f"Deployment failed: {deploy_result['message']}")
```

### CI/CD Integration

Create `.github/workflows/docs.yml`:

```yaml
name: Build and Deploy Documentation

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build-docs:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pyyaml jinja2

    - name: Build documentation
      run: |
        python -c "
        from docs.framework.deployment.builder import DocumentationBuilder
        builder = DocumentationBuilder('docs/config.yaml')
        report = builder.build_documentation()
        print(f'Built {report[\"modules_processed\"]} modules')
        "

    - name: Deploy to GitHub Pages
      if: github.ref == 'refs/heads/main'
      run: |
        python -c "
        from docs.framework.deployment.deployer import DocumentationDeployer
        from pathlib import Path
        config = {'deployment': {'github_pages': True}}
        deployer = DocumentationDeployer(config)
        result = deployer.deploy_to_github_pages(Path('docs/generated/html'))
        print(f'Deployment: {result[\"success\"]}')
        "
```

## Best Practices

### Writing Documentation

1. **Use descriptive docstrings**:
```python
def process_data(data: List[Dict[str, Any]],
                 filter_func: Optional[Callable] = None) -> List[Dict[str, Any]]:
    """Process a list of data dictionaries with optional filtering.

    This function takes raw data dictionaries and applies optional
    filtering logic to produce a cleaned dataset suitable for analysis.

    Args:
        data: List of dictionaries containing raw data records
        filter_func: Optional function to filter data records.
                    Should return True for records to keep.

    Returns:
        List of processed and optionally filtered data dictionaries

    Raises:
        ValueError: If data is empty or contains invalid records
        TypeError: If filter_func is not callable

    Examples:
        >>> data = [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]
        >>> result = process_data(data)
        >>> len(result)
        2

        >>> adults_only = lambda x: x['age'] >= 18
        >>> adults = process_data(data, adults_only)
        >>> all(person['age'] >= 18 for person in adults)
        True
    """
```

2. **Include examples in docstrings**
3. **Document all parameters and return values**
4. **Use type hints consistently**
5. **Keep docstrings up to date with code changes**

### Template Development

1. **Use semantic HTML structure**
2. **Make templates responsive**
3. **Include proper meta tags**
4. **Use consistent styling**
5. **Test templates with various content**

### Quality Maintenance

1. **Run validation regularly**
2. **Set coverage thresholds**
3. **Review quality reports**
4. **Fix broken links promptly**
5. **Update documentation with code changes**

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure the docs framework is in your Python path
2. **Template not found**: Check template directory configuration
3. **Sandbox security errors**: Review blocked operations in sandbox configuration
4. **Link validation timeouts**: Increase timeout or add domains to exclusion list
5. **Coverage calculation errors**: Check for syntax errors in source files

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Framework components will now show debug information
```

### Performance Tips

1. **Use selective parsing**: Only include necessary source directories
2. **Cache templates**: Template engine caches compiled templates
3. **Limit concurrent link validation**: Reduce max_concurrent for slower connections
4. **Use format-specific generation**: Only generate needed output formats
5. **Enable compression**: Use gzip for deployed documentation

## Framework Architecture

```
docs/framework/
├── generators/
│   ├── ast_parser.py          # Python code analysis
│   ├── rust_parser.py         # Rust code analysis (stub)
│   └── template_engine.py     # Jinja2 templating
├── validation/
│   ├── coverage_analyzer.py   # Documentation coverage
│   ├── quality_checker.py     # Quality assessment
│   └── link_validator.py      # Link validation
├── server/
│   ├── app.py                 # FastAPI server
│   └── sandbox.py             # Code execution sandbox
└── deployment/
    ├── builder.py             # Documentation builder
    └── deployer.py            # GitHub Pages deployer
```

The framework follows a modular design where each component can be used independently or together as part of the complete documentation workflow.

## Contributing

To extend the framework:

1. **Add new parsers**: Implement parsers for other languages in `generators/`
2. **Create custom validators**: Add validation logic in `validation/`
3. **Build new templates**: Add templates in `templates/`
4. **Extend deployment**: Add new deployment targets in `deployment/`
5. **Improve sandbox**: Enhance code execution security and features

Each component includes comprehensive unit tests and follows the project's coding standards.