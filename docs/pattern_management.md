# Pattern Management System

The PatternManager provides centralized pattern management for file filtering and ecosystem detection in workspace-qdrant-mcp. It loads hardcoded patterns from embedded YAML files and allows customization through user-defined patterns.

## Overview

The PatternManager system consists of:

- **Hardcoded Patterns**: Embedded YAML files with comprehensive patterns for common file types and ecosystems
- **Custom Patterns**: User-defined patterns that take precedence over hardcoded ones
- **Pattern Matching**: Glob-based pattern matching with caching for performance
- **Ecosystem Detection**: Project type detection based on indicator files
- **Language Information**: File extension to language mapping

## Architecture

```
PatternManager
├── Hardcoded Patterns (from YAML files)
│   ├── include_patterns.yaml    # Files to include in processing
│   ├── exclude_patterns.yaml    # Files to exclude from processing
│   ├── project_indicators.yaml  # Ecosystem detection rules
│   └── language_extensions.yaml # Language detection mappings
└── Custom Patterns (from configuration)
    ├── custom_include_patterns   # User-defined include patterns
    ├── custom_exclude_patterns   # User-defined exclude patterns
    └── custom_project_indicators # User-defined ecosystem rules
```

## Pattern Files Structure

### Include Patterns (`include_patterns.yaml`)

Defines which files should be included in processing:

```yaml
programming:
  - pattern: "*.py"
    category: "python"
    priority: "high"
  - pattern: "*.js"
    category: "javascript"
    priority: "medium"

documentation:
  - "*.md"
  - "*.rst"
  - "*.txt"

configuration:
  - pattern: "*.yaml"
    category: "config"
  - pattern: "*.json"
    category: "config"
```

### Exclude Patterns (`exclude_patterns.yaml`)

Defines which files should be excluded from processing:

```yaml
build_artifacts:
  - "*.o"
  - "*.so"
  - "*.dylib"
  - "__pycache__/**"
  - "target/**"

version_control:
  - ".git/**"
  - ".svn/**"
  - "*.orig"
  - "*.rej"

temporary:
  - pattern: "*.tmp"
    category: "temp"
  - pattern: "*.swp"
    category: "editor"
```

### Project Indicators (`project_indicators.yaml`)

Defines ecosystem detection rules:

```yaml
ecosystems:
  python:
    required_files:
      - "setup.py"
    optional_files:
      - "requirements.txt"
      - "pyproject.toml"
      - "Pipfile"
    min_optional_files: 1

  rust:
    required_files:
      - "Cargo.toml"
    optional_files:
      - "Cargo.lock"
      - "rust-toolchain.toml"
    min_optional_files: 0

  javascript:
    required_files:
      - "package.json"
    optional_files:
      - "yarn.lock"
      - "package-lock.json"
    min_optional_files: 0
```

### Language Extensions (`language_extensions.yaml`)

Maps file extensions to programming languages:

```yaml
programming_languages:
  python:
    extensions: [".py", ".pyx", ".pyi"]
    ecosystem: "python"

  rust:
    extensions: [".rs"]
    ecosystem: "rust"

  javascript:
    extensions: [".js", ".mjs", ".jsx"]
    ecosystem: "javascript"

markup_languages:
  markdown:
    extensions: [".md", ".markdown", ".mdown"]
    category: "documentation"

  yaml:
    extensions: [".yaml", ".yml"]
    category: "configuration"
```

## Usage

### Basic Usage

```python
from common.core.pattern_manager import PatternManager

# Initialize with default hardcoded patterns
pattern_manager = PatternManager()

# Check if a file should be included
should_include, reason = pattern_manager.should_include("main.py")
print(f"Include: {should_include}, Reason: {reason}")

# Check if a file should be excluded
should_exclude, reason = pattern_manager.should_exclude("__pycache__/module.pyc")
print(f"Exclude: {should_exclude}, Reason: {reason}")

# Detect project ecosystem
ecosystems = pattern_manager.detect_ecosystem("/path/to/project")
print(f"Detected ecosystems: {ecosystems}")

# Get language information
lang_info = pattern_manager.get_language_info("script.py")
print(f"Language info: {lang_info}")
```

### Custom Patterns

```python
# Initialize with custom patterns
pattern_manager = PatternManager(
    custom_include_patterns=["*.custom", "*.special"],
    custom_exclude_patterns=["*.ignore", "temp/**"],
    custom_project_indicators={
        "my_ecosystem": {
            "required_files": ["my_config.conf"],
            "optional_files": ["my_lock.lock"],
            "min_optional_files": 0
        }
    }
)

# Custom patterns take precedence over hardcoded ones
should_include, reason = pattern_manager.should_include("file.custom")
print(f"Include: {should_include}, Reason: {reason}")
# Output: Include: True, Reason: custom_include_pattern: *.custom
```

### Configuration Integration

```python
# With WorkspaceConfig (future integration)
from workspace_qdrant_mcp.core.config import WorkspaceConfig

config = WorkspaceConfig(
    custom_include_patterns=["*.workflow", "*.pipeline"],
    custom_exclude_patterns=["*.cache", "build/**"]
)

pattern_manager = PatternManager(
    custom_include_patterns=config.custom_include_patterns,
    custom_exclude_patterns=config.custom_exclude_patterns
)
```

## Pattern Matching Rules

### Precedence

1. **Custom Include Patterns** (highest precedence)
2. **Hardcoded Include Patterns**
3. **Custom Exclude Patterns**
4. **Hardcoded Exclude Patterns** (lowest precedence)

### Pattern Syntax

The PatternManager uses glob patterns with these features:

- `*` - Matches any characters except path separators
- `**` - Matches directories recursively
- `?` - Matches any single character
- `[seq]` - Matches any character in seq
- `[!seq]` - Matches any character not in seq

Examples:
- `*.py` - All Python files
- `src/**/*.js` - All JavaScript files in src directory (recursive)
- `test_*.py` - All Python test files
- `build/**` - All files in build directory (recursive)
- `*.{py,js,rs}` - Files with multiple extensions (if supported)

### Directory Patterns

- Patterns ending with `/` match directories
- Use `**` for recursive directory matching
- Examples:
  - `__pycache__/` - Match __pycache__ directories
  - `target/**` - Match everything in target directories
  - `node_modules/**` - Match everything in node_modules

## Performance Features

### Caching

PatternManager caches pattern matching results for performance:

```python
# Cache management
pattern_manager.clear_cache()  # Clear pattern cache

# Cache statistics
stats = pattern_manager.get_statistics()
print(f"Cache size: {stats['cache']['size']}")
print(f"Cache limit: {stats['cache']['limit']}")
```

### Statistics

Get comprehensive statistics about loaded patterns:

```python
stats = pattern_manager.get_statistics()
print(f"Include patterns: {stats['include_patterns']}")
print(f"Exclude patterns: {stats['exclude_patterns']}")
print(f"Project indicators: {stats['project_indicators']}")
print(f"Language extensions: {stats['language_extensions']}")
```

## Ecosystem Detection

The PatternManager can detect project ecosystems based on indicator files:

```python
# Detect ecosystems in a project directory
ecosystems = pattern_manager.detect_ecosystem("/path/to/project")

# Example output: ['python', 'javascript'] for a mixed project
```

### Detection Rules

Each ecosystem has:
- **Required Files**: Must have at least one
- **Optional Files**: Additional indicators
- **Minimum Optional**: Minimum number of optional files needed

## Language Information

Get detailed language information for files:

```python
lang_info = pattern_manager.get_language_info("script.py")
# Returns: {
#     'language': 'python',
#     'extension': '.py',
#     'file_path': 'script.py'
# }
```

## Error Handling

PatternManager handles errors gracefully:

- **Missing Pattern Files**: Falls back to empty patterns
- **Invalid YAML**: Logs error and continues with empty patterns
- **Pattern Matching Errors**: Returns False for failed matches
- **Invalid Paths**: Handles edge cases without crashing

## Best Practices

### Custom Pattern Design

1. **Be Specific**: Avoid overly broad patterns that might match unintended files
2. **Use Categories**: Group related patterns for better organization
3. **Test Patterns**: Verify your patterns match the intended files
4. **Document Purpose**: Include comments in YAML files explaining pattern purposes

### Performance Optimization

1. **Pattern Order**: Put most common patterns first
2. **Cache Awareness**: Reuse PatternManager instances when possible
3. **Batch Operations**: Process multiple files with the same instance

### Integration Guidelines

1. **Custom Patterns**: Always prefer custom patterns for project-specific needs
2. **Configuration**: Integrate with WorkspaceConfig for user customization
3. **Validation**: Validate custom patterns before passing to PatternManager
4. **Fallbacks**: Always provide sensible defaults for missing custom patterns

## Examples

### Python Project Configuration

```python
python_patterns = PatternManager(
    custom_include_patterns=[
        "*.py", "*.pyi", "*.pyx",  # Python files
        "*.yaml", "*.yml",         # Configuration files
        "*.md", "*.rst",           # Documentation
        "requirements*.txt",       # Requirements files
        "setup.py", "setup.cfg",   # Setup files
        "pyproject.toml"           # Modern Python config
    ],
    custom_exclude_patterns=[
        "__pycache__/**",          # Python cache
        "*.pyc", "*.pyo",         # Compiled Python
        ".pytest_cache/**",       # Pytest cache
        "build/**", "dist/**",    # Build artifacts
        ".venv/**", "venv/**",    # Virtual environments
        "*.egg-info/**"           # Egg info
    ]
)
```

### JavaScript/TypeScript Project

```python
js_patterns = PatternManager(
    custom_include_patterns=[
        "*.js", "*.jsx", "*.ts", "*.tsx",  # JS/TS files
        "*.json",                          # Config files
        "*.md"                             # Documentation
    ],
    custom_exclude_patterns=[
        "node_modules/**",                 # Dependencies
        "dist/**", "build/**",             # Build output
        "coverage/**",                     # Coverage reports
        ".next/**", ".nuxt/**",            # Framework builds
        "*.min.js", "*.bundle.js"          # Minified files
    ]
)
```

### Multi-Language Project

```python
polyglot_patterns = PatternManager(
    custom_include_patterns=[
        # Source code
        "*.py", "*.js", "*.rs", "*.go", "*.java",
        # Configuration
        "*.yaml", "*.json", "*.toml",
        # Documentation
        "*.md", "*.rst", "*.txt"
    ],
    custom_exclude_patterns=[
        # Build artifacts (all languages)
        "**/target/**", "**/build/**", "**/dist/**",
        # Cache directories
        "**/__pycache__/**", "**/node_modules/**",
        # IDE files
        "**/.vscode/**", "**/.idea/**"
    ]
)
```

This pattern management system provides flexible, powerful file filtering capabilities that can be customized for any project's needs while maintaining good performance through caching and efficient pattern matching.