# Contributing to workspace-qdrant-mcp

We welcome contributions to workspace-qdrant-mcp! This project follows a test-driven development approach with comprehensive quality gates.

## Development Setup

### Prerequisites

- **Python 3.9+** with pip or uv
- **Git** for version control
- **Docker** for Qdrant server (recommended)
- **Qdrant server** running locally or remotely

### Clone and Setup

```bash
# Clone repository
git clone https://github.com/ChrisGVE/workspace-qdrant-mcp.git
cd workspace-qdrant-mcp

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
pip install -e .[dev]

# Or using uv (recommended)
uv sync --dev
```

### Start Development Qdrant Server

**Using Docker:**
```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

**Using Docker Compose:**
```bash
# Create docker-compose.yml if not exists
cat > docker-compose.yml << EOF
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - ./qdrant_storage:/qdrant/storage
EOF

# Start Qdrant
docker-compose up -d qdrant
```

### Development Environment

Create `.env` file for development:
```bash
# Copy example configuration
cp .env.example .env

# Edit for development
cat > .env << EOF
# Qdrant Configuration
QDRANT_URL=http://localhost:6333

# Development settings
WORKSPACE_QDRANT_DEBUG=true
WORKSPACE_QDRANT_HOST=127.0.0.1
WORKSPACE_QDRANT_PORT=8000

# Optional: GitHub user for testing
GITHUB_USER=your-username
EOF
```

### Verify Development Setup

```bash
# Validate configuration
workspace-qdrant-validate --verbose

# Start development server
workspace-qdrant-mcp --debug

# Test basic functionality
curl http://localhost:8000/health
```

## Quality Gates

All contributions must pass our comprehensive quality gates:

### 1. Code Formatting and Linting

```bash
# Format code with Black
black src/ tests/

# Lint with Ruff
ruff check src/ tests/ --fix

# Type checking with mypy
mypy src/
```

**Pre-commit hooks (recommended):**
```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### 2. Testing Requirements

**Minimum 80% test coverage required for all new code.**

```bash
# Run full test suite
pytest

# Run with coverage report
pytest --cov=src/workspace_qdrant_mcp --cov-report=html --cov-report=term-missing

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m e2e          # End-to-end tests only
pytest -m slow         # Long-running tests
```

**Test categories:**
- **Unit tests**: Fast, isolated, no external dependencies
- **Integration tests**: Test component interactions
- **End-to-end tests**: Full workflow testing
- **Performance tests**: Benchmark validations

### 3. Performance Benchmarks

**Required performance thresholds must be maintained:**

```bash
# Run performance benchmarks
pytest tests/benchmarks/ --benchmark-only

# Specific benchmark categories
pytest tests/benchmarks/test_search_performance.py
pytest tests/benchmarks/test_embedding_performance.py
```

**Performance requirements:**
- Symbol search: ≥90% precision, ≥70% recall
- Semantic search: ≥84% precision, ≥70% recall
- Hybrid search: ≥90% precision, ≥75% recall
- Response time: <100ms average for search operations
- Memory usage: <200MB RSS during normal operations

### 4. Integration Tests

```bash
# Test with real Qdrant instance
pytest -m requires_qdrant

# Test project detection with Git
pytest -m requires_git

# Test MCP protocol compliance
pytest -m mcp_protocol
```

## Development Workflow

### 1. Create Feature Branch

```bash
# Create and checkout feature branch
git checkout -b feature/your-feature-name

# For bug fixes
git checkout -b fix/issue-description

# For documentation
git checkout -b docs/update-description
```

### 2. Test-Driven Development

**Write tests first:**
```bash
# Create test file
touch tests/test_your_feature.py

# Write failing tests
vim tests/test_your_feature.py
```

**Example test structure:**
```python
import pytest
from workspace_qdrant_mcp.your_module import YourClass

class TestYourFeature:
    def test_basic_functionality(self):
        """Test the happy path."""
        instance = YourClass()
        result = instance.your_method("input")
        assert result == "expected_output"
    
    def test_error_handling(self):
        """Test error conditions."""
        instance = YourClass()
        with pytest.raises(ValueError, match="specific error message"):
            instance.your_method("invalid_input")
    
    @pytest.mark.asyncio
    async def test_async_functionality(self):
        """Test async operations."""
        instance = YourClass()
        result = await instance.async_method()
        assert result is not None
```

**Implement feature:**
```bash
# Create implementation
vim src/workspace_qdrant_mcp/your_module.py

# Run tests continuously during development
pytest --watch tests/test_your_feature.py
```

### 3. Code Quality Validation

**Run complete quality check:**
```bash
# Format and lint
black src/ tests/
ruff check src/ tests/ --fix

# Type checking
mypy src/

# Full test suite
pytest

# Performance validation
pytest tests/benchmarks/ --benchmark-only
```

**Quality check script:**
```bash
#!/bin/bash
# scripts/quality-check.sh

set -e

echo "🧹 Formatting code..."
black src/ tests/

echo "🔍 Linting..."
ruff check src/ tests/ --fix

echo "🏷️  Type checking..."
mypy src/

echo "🧪 Running tests..."
pytest --cov=src/workspace_qdrant_mcp --cov-fail-under=80

echo "⚡ Performance benchmarks..."
pytest tests/benchmarks/ --benchmark-only

echo "✅ All quality checks passed!"
```

### 4. Performance Testing

**Benchmark your changes:**
```bash
# Create performance baseline
pytest tests/benchmarks/ --benchmark-save=before

# Make your changes...

# Compare performance
pytest tests/benchmarks/ --benchmark-compare=before
```

**Performance regression detection:**
```python
# tests/benchmarks/test_your_feature.py
import pytest

class TestYourFeaturePerformance:
    @pytest.mark.benchmark(group="search")
    def test_search_performance(self, benchmark):
        """Ensure search performance meets requirements."""
        def search_operation():
            # Your search implementation
            return perform_search("test query")
        
        result = benchmark(search_operation)
        
        # Assert performance requirements
        assert benchmark.stats.mean < 0.1  # <100ms average
        assert result["precision"] >= 0.90  # ≥90% precision
```

## Code Standards

### Code Style

- **Follow PEP 8** (enforced by Black)
- **Line length: 88 characters** (Black default)
- **Type hints required** for all public APIs
- **Docstrings required** for all public functions/classes

### Documentation Standards

**Function documentation:**
```python
def search_workspace(
    query: str, 
    mode: str = "hybrid", 
    limit: int = 10
) -> Dict[str, Any]:
    """Search across workspace collections.
    
    Performs intelligent search using the specified mode across all
    workspace collections with optional result limiting.
    
    Args:
        query: The search query text
        mode: Search mode - "hybrid", "semantic", "exact", or "symbol"
        limit: Maximum number of results to return
        
    Returns:
        Dictionary containing search results and metadata:
        - results: List of matching documents
        - search_stats: Performance and relevance metrics
        
    Raises:
        ValueError: If mode is not supported
        QdrantConnectionError: If Qdrant server is unavailable
        
    Example:
        >>> results = search_workspace("vector similarity", "hybrid", 10)
        >>> print(f"Found {len(results['results'])} matches")
    """
```

**Class documentation:**
```python
class WorkspaceManager:
    """Manages workspace detection and collection organization.
    
    Handles automatic project detection from Git repositories,
    collection naming conventions, and workspace-scoped operations.
    
    Attributes:
        project_name: Detected project name from Git
        collections: List of managed collections
        qdrant_client: Connected Qdrant client instance
    """
```

### Testing Standards

**Test organization:**
```
tests/
├── unit/                    # Fast, isolated tests
│   ├── test_search.py
│   └── test_workspace.py
├── integration/             # Component interaction tests
│   ├── test_mcp_tools.py
│   └── test_qdrant_integration.py
├── e2e/                    # End-to-end workflow tests
│   └── test_complete_workflows.py
├── benchmarks/             # Performance tests
│   └── test_search_performance.py
└── conftest.py             # Shared fixtures
```

**Test naming conventions:**
- `test_<functionality>` for basic tests
- `test_<functionality>_with_<condition>` for specific scenarios
- `test_<functionality>_raises_<exception>` for error cases

**Required test coverage:**
- **New features: 90%+ coverage**
- **Bug fixes: Test the specific bug**
- **Refactoring: Maintain existing coverage**
- **Critical paths: 100% coverage**

## Contribution Types

### High Priority Contributions

**Core functionality:**
- Advanced search algorithms and optimizations
- New embedding model integrations
- Enhanced metadata filtering capabilities
- Performance optimizations for large collections

**Developer experience:**
- CLI tool improvements
- Better error messages and debugging
- Documentation and examples
- Integration guides

### Medium Priority Contributions

**Platform support:**
- Windows compatibility improvements
- macOS optimization
- Docker/Kubernetes deployment guides
- CI/CD pipeline enhancements

**Integrations:**
- Additional vector database backends
- Alternative embedding providers
- MCP client library improvements
- VS Code extension enhancements

### Documentation & Testing

**Always welcome:**
- Tutorial content and real-world examples
- API documentation improvements
- Integration test scenarios
- Performance benchmarking scripts
- Error handling edge cases

## Submission Process

### Pull Request Requirements

**Before submitting:**
- [ ] All quality gates pass
- [ ] Performance benchmarks within thresholds
- [ ] Documentation updated
- [ ] Tests cover new functionality
- [ ] No security vulnerabilities introduced

**PR description template:**
```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance optimization
- [ ] Code refactoring

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Performance benchmarks meet requirements
- [ ] Manual testing completed

## Performance Impact
Describe any performance implications:
- Execution time changes: ±X ms
- Memory usage changes: ±X MB
- Benchmark results: [attach if applicable]

## Breaking Changes
List any breaking changes and migration steps required.

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] Performance validated
```

### Review Process

**Requirements for approval:**
- [ ] All automated checks pass
- [ ] Code review by maintainer
- [ ] Performance impact assessed
- [ ] Documentation review completed
- [ ] No security issues identified

**Review timeline:**
- **Initial review:** Within 48 hours
- **Follow-up reviews:** Within 24 hours
- **Critical fixes:** Expedited review

**Review criteria:**
- Code quality and maintainability
- Test coverage and quality
- Performance impact
- Documentation completeness
- Adherence to project standards

## Release Process

### Version Management

workspace-qdrant-mcp follows [Semantic Versioning](https://semver.org/):

- **MAJOR.MINOR.PATCH** (e.g., 1.2.3)
- **Major:** Breaking changes
- **Minor:** New features (backward compatible)
- **Patch:** Bug fixes (backward compatible)

### Conventional Commits

Use [Conventional Commits](https://www.conventionalcommits.org/) for automatic changelog generation:

```bash
# Features
git commit -m "feat: add advanced metadata filtering"

# Bug fixes
git commit -m "fix: resolve memory leak in embedding processing"

# Performance improvements
git commit -m "perf: optimize hybrid search ranking algorithm"

# Documentation
git commit -m "docs: add API examples for search operations"

# Breaking changes
git commit -m "feat!: redesign collection naming convention"
```

### Development Branches

- **`main`:** Stable releases only
- **`develop`:** Integration branch for features
- **`feature/*`:** Individual feature development
- **`fix/*`:** Bug fixes
- **`docs/*`:** Documentation updates

### Release Pipeline

**Automated testing on PR:**
- Unit and integration tests
- Performance benchmarks
- Security vulnerability scanning
- Multi-platform compatibility testing

**Release creation:**
- Automatic version bumping
- Changelog generation
- PyPI package publication
- Docker image builds
- GitHub release creation

## Getting Help

### Development Questions

- **GitHub Discussions:** For design questions and feature discussions
- **GitHub Issues:** For bug reports and specific problems
- **Code Comments:** For implementation questions during review

### Debugging

**Enable debug logging:**
```bash
workspace-qdrant-mcp --debug
```

**Validate configuration:**
```bash
workspace-qdrant-validate --verbose --fix
```

**Check logs:**
```bash
tail -f ~/.local/state/workspace-qdrant-mcp/logs/server.log
```

### Community

- **Respectful communication:** Follow our Code of Conduct
- **Constructive feedback:** Focus on code, not individuals
- **Knowledge sharing:** Help others learn and contribute
- **Recognition:** Contributors are highlighted in releases

## Code of Conduct

workspace-qdrant-mcp follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you agree to uphold this code.

### Our Standards

**Positive behavior:**
- Using welcoming and inclusive language
- Respecting different viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community

**Unacceptable behavior:**
- Harassment, discrimination, or hostile behavior
- Publishing private information without permission
- Trolling, insulting comments, or personal attacks
- Any conduct inappropriate in a professional setting

Thank you for contributing to workspace-qdrant-mcp! Your efforts help make this project better for everyone.